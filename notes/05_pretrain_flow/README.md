# 📍 Day 5: 预训练实战（Pretrain）——从零让模型学会接龙

## 🎯 今日目标
1. [ ] 理解自回归预训练的本质任务：**Next-Token Prediction（下一个词预测）**
   * 为什么预训练的目标是最大化文本语料的对数似然？
2. [ ] 攻克输入与标签的错位对齐逻辑：
   * 输入：`input_ids = [A, B, C, D]`
   * 标签：`target_ids = [B, C, D, E]`
   * 代码中的位移操作：`shift_logits = logits[..., :-1, :]` vs `shift_labels = labels[..., 1:]`
3. [ ] 剖析现代 LLM 训练循环的四大法宝：
   * **AdamW 优化器**：带解耦权重衰减（Decoupled Weight Decay）的自适应动量优化。
   * **学习率调度（LR Schedule）**：Linear Warmup（预热）+ Cosine Annealing（余弦退火衰减）。
   * **梯度裁剪（Gradient Clipping）**：`clip_grad_norm_` 防止梯度爆炸导致参数变为 NaN。
   * **梯度累积（Gradient Accumulation）**：显存有限时，累积多步反向传播梯度再执行 `optimizer.step()`，等效放大 Batch Size。
4. [ ] 动手实战：跑通极小步数的预训练测试，观察 Loss 从高位（约 8~9）平稳下降。

## 🔍 涉及文件
* [dataset/lm_dataset.py](file:///Users/kevin/Code/github/AI/minimind/dataset/lm_dataset.py)（`PretrainDataset` 类）
* [trainer/train_pretrain.py](file:///Users/kevin/Code/github/AI/minimind/trainer/train_pretrain.py)（预训练主循环）
* [trainer/trainer_utils.py](file:///Users/kevin/Code/github/AI/minimind/trainer/trainer_utils.py)（学习率计算、种子设置等工具）

## 🧪 今日实验指引
* `exp1_pretrain_dataset_inspect.py`：读取一行真实数据（`dataset/pretrain_t2t_mini.jsonl`），打印其 tokenize 结果与 input/label 错位切片。
* `exp2_lr_scheduler_curve.py`：编写脚本模拟 Warmup + Cosine 退火，绘制学习率随 step 变化的曲线。
* `exp3_mini_pretrain_run.py`：启动微型预训练任务（跑 50 个 steps），打印实时 Loss 变化。
