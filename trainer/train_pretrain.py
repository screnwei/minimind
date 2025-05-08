import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    # 定义损失函数，这里使用交叉熵损失，且不做reduction，方便后续加权
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()  # 记录本轮开始时间

    # 遍历训练集的每个batch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)  # 输入数据转到指定设备
        Y = Y.to(args.device)  # 标签转到指定设备
        loss_mask = loss_mask.to(args.device)  # 损失掩码转到指定设备

        # 计算当前步的学习率（余弦退火）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新优化器的学习率

        # 自动混合精度上下文
        # with 是 Python 的上下文管理器（Context Manager）语法。
        # 它用于简化资源管理，比如文件读写、锁、数据库连接、自动混合精度等场景。
        # with 语句可以确保在代码块执行完毕后，自动进行清理操作（如关闭文件、释放资源等），即使发生异常也能保证资源被正确释放。
        with ctx:
            res = model(X)  # 前向传播
            # 计算逐token损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            # 只对有效token计算损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 加入辅助损失
            loss += res.aux_loss
            # 梯度累积，损失缩放
            loss = loss / args.accumulation_steps

        # 反向传播（自动混合精度缩放）
        scaler.scale(loss).backward()

        # 梯度累积步数达到设定值时，进行参数更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)  # 优化器更新
            scaler.update()         # 更新scaler

            optimizer.zero_grad(set_to_none=True)  # 梯度清零

        # 日志打印
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # wandb日志记录
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 模型定期保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            # 兼容DDP和单卡
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 半精度保存模型权重
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

# 预训练代码
# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")  # 输出目录，保存模型和日志等文件
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=32)  # 每个batch的样本数
    parser.add_argument("--learning_rate", type=float, default=5e-4)  # 学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型（如bfloat16、float16等）
    parser.add_argument("--use_wandb", action="store_true")  # 是否启用wandb日志
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")  # wandb项目名称
    parser.add_argument("--num_workers", type=int, default=1)  # DataLoader加载数据的线程数
    parser.add_argument("--ddp", action="store_true")  # 是否使用分布式训练（DDP）
    parser.add_argument("--accumulation_steps", type=int, default=8)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 学习率预热步数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志打印间隔（步）
    parser.add_argument("--save_interval", type=int, default=100)  # 模型保存间隔（步）
    parser.add_argument('--local_rank', type=int, default=-1)  # DDP本地rank
    parser.add_argument('--hidden_size', default=512, type=int)  # 模型隐藏层维度
    parser.add_argument('--num_hidden_layers', default=8, type=int)  # Transformer层数
    parser.add_argument('--max_seq_len', default=512, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用MoE结构
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")  # 预训练数据路径
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
