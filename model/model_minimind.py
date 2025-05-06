# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout  # dropout 比率，用于防止过拟合
        self.bos_token_id = bos_token_id  # 序列开始标记的 token ID
        self.eos_token_id = eos_token_id  # 序列结束标记的 token ID
        self.hidden_act = hidden_act  # 隐藏层激活函数类型
        self.hidden_size = hidden_size  # 隐藏层维度大小
        self.intermediate_size = intermediate_size  # 中间层维度大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码长度
        self.num_attention_heads = num_attention_heads  # 注意力头数量
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数量
        self.num_key_value_heads = num_key_value_heads  # 键值注意力头数量
        self.vocab_size = vocab_size  # 词表大小
        self.rms_norm_eps = rms_norm_eps  # RMS 归一化的 epsilon 值
        self.rope_theta = rope_theta  # RoPE 旋转位置编码的 theta 参数
        self.flash_attn = flash_attn  # 是否使用 Flash Attention 优化
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）是一种归一化方法。
    它对每个样本的最后一个维度做均方根归一化，并带有可学习缩放参数。
    优点是实现简单、无偏置、数值稳定性好，常用于Transformer等模型。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 防止除零的小常数，提升数值稳定性
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放参数

    def _norm(self, x):
        # 计算均方根归一化：x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先做归一化，再乘以可学习缩放参数
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Attention 实现了多头自注意力机制，支持Flash Attention加速。
    支持分离的key/value头（可用于多query attention），并支持缓存机制。
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # key/value头数量，若未指定则与注意力头数量一致
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 确保注意力头数量能被key/value头整除
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # 本地注意力头数量
        self.n_local_heads = args.num_attention_heads
        # 本地key/value头数量
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个key/value头被复制的次数（多query attention）
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads
        # 查询（Q）投影层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        # 键（K）投影层
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 值（V）投影层
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影层
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        # 注意力权重dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 是否支持Flash Attention（高效attention实现）
        self.flash = (hasattr(torch.nn.functional, 'scaled_dot_product_attention') 
                     and args.flash_attn 
                     and torch.cuda.is_available())

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播：
        1. 输入x经过Q/K/V线性变换，reshape为多头格式。
        2. 应用RoPE旋转位置编码。
        3. 若有past_key_value则拼接历史K/V，实现自回归缓存。
        4. K/V复制扩展到所有Q头（多query attention）。
        5. 支持Flash Attention加速，否则用常规softmax注意力。
        6. 输出加残差dropout和线性投影。
        Args:
            x: 输入隐藏状态 (batch, seq_len, hidden_size)
            position_embeddings: 位置编码（cos, sin）
            past_key_value: 历史K/V缓存
            use_cache: 是否返回新的K/V缓存
            attention_mask: 注意力掩码
        Returns:
            output: 注意力输出 (batch, seq_len, hidden_size)
            past_kv: 新的K/V缓存
        """
        bsz, seq_len, _ = x.shape
        # Q/K/V线性变换并reshape为多头格式
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用RoPE旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 拼接历史K/V缓存（自回归生成）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # K/V复制扩展到所有Q头（多query attention）
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            # Flash Attention高效实现
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            # 常规softmax注意力
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 下三角mask，保证自回归
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                # 扩展attention mask到多头
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 恢复shape并做输出投影和dropout
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    FeedForward（前馈神经网络）是Transformer中的MLP子层。
    结构为：线性变换 -> 激活函数 -> 线性变换 -> Dropout。
    这里采用SwiGLU结构（门控激活），提升模型表达能力。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 若未指定中间层维度，自动按hidden_size*8/3并向上取整到64的倍数
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 门控投影（SwiGLU结构）
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 下投影，将中间层还原为hidden_size
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 上投影，提升维度
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数（如silu）

    def forward(self, x):
        """
        前向传播：
        1. 输入x先经过门控投影（gate_proj），再经过激活函数。
        2. 与上投影（up_proj）结果做逐元素乘法（SwiGLU结构）。
        3. 结果经过下投影（down_proj）还原维度。
        4. 最后做Dropout。
        Args:
            x: 输入隐藏状态 (batch, seq_len, hidden_size)
        Returns:
            输出 (batch, seq_len, hidden_size)
        """
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    MoEGate（门控网络）用于为每个token选择最合适的专家（Expert）。
    通过门控权重决定token分配到哪些专家，并可计算辅助损失（aux_loss）优化专家分配。
    支持softmax门控、top-k选择、概率归一化和多种辅助损失。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 专家总数

        self.scoring_func = config.scoring_func  # 门控分数函数（如softmax）
        self.alpha = config.aux_loss_alpha  # 辅助损失系数
        self.seq_aux = config.seq_aux  # 是否序列级别辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否对top-k概率归一化
        self.gating_dim = config.hidden_size  # 门控输入维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 门控权重参数
        self.reset_parameters()  # 权重初始化

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Kaiming初始化

    def forward(self, hidden_states):
        """
        前向传播：
        1. 计算每个token分配到各专家的门控分数。
        2. 选择top-k分数最大的专家，并获得其分配概率。
        3. 可选：对top-k概率归一化。
        4. 训练时计算辅助损失，优化专家分配的均匀性和多样性。
        Args:
            hidden_states: 输入隐藏状态 (batch, seq_len, hidden_size)
        Returns:
            topk_idx: 每个token分配的top-k专家编号 (batch*seq_len, top_k)
            topk_weight: 每个token分配给top-k专家的概率 (batch*seq_len, top_k)
            aux_loss: 辅助损失（训练时用于优化专家分配）
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        # 计算门控分数（logits），形状：(batch*seq_len, n_routed_experts)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择top-k专家及其分配概率
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 可选：对top-k概率归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 训练时计算辅助损失
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # 序列级别辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # token级别辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    MOEFeedForward 实现了门控专家（Mixture of Experts, MoE）前馈网络。
    每个token通过门控机制动态选择若干专家（FeedForward子网络）进行处理，提升模型容量和表达能力。
    支持共享专家（shared experts）和辅助损失（aux_loss）以优化专家分配。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 专家网络列表，每个专家是一个FeedForward子网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 门控网络，决定每个token分配给哪些专家
        self.gate = MoEGate(config)
        # 共享专家（可选），所有token都经过
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        前向传播：
        1. 通过门控网络为每个token选择top-k个专家及其权重。
        2. 将token分配给对应专家，专家独立处理后按权重加权求和。
        3. 若有共享专家，所有token再经过共享专家并累加。
        4. 训练时返回辅助损失（aux_loss），用于优化专家分配。
        Args:
            x: 输入隐藏状态 (batch, seq_len, hidden_size)
        Returns:
            y: MoE前馈输出 (batch, seq_len, hidden_size)
        """
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练时：每个token复制num_experts_per_tok份，分配给top-k专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            # 按专家权重加权求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理时：高效批量分配token到专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        # 共享专家处理
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss  # 保存辅助损失
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理时高效分配token到专家，避免重复计算。
        Args:
            x: 展平后的输入 (num_tokens, hidden_size)
            flat_expert_indices: 每个token分配的专家编号 (num_tokens,)
            flat_expert_weights: 每个token分配的专家权重 (num_tokens, 1)
        Returns:
            expert_cache: 所有token的MoE输出 (num_tokens, hidden_size)
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 按专家分组批量处理token，提升推理效率
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    MiniMindBlock 表示 Transformer 的一个基本块（Block），包含自注意力层、前馈网络（MLP）和归一化层。
    每个Block的结构为：LayerNorm -> Attention -> 残差 -> LayerNorm -> MLP -> 残差。
    支持普通前馈和门控专家（MoE）结构。
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        # 注意力头数量
        self.num_attention_heads = config.num_attention_heads
        # 隐藏层维度
        self.hidden_size = config.hidden_size
        # 每个注意力头的维度
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 自注意力层
        self.self_attn = Attention(config)

        self.layer_id = layer_id  # 当前Block的层编号
        # 输入归一化层（RMSNorm），用于Attention前
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Attention后归一化层（RMSNorm），用于MLP前
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 前馈网络（MLP），可为普通FeedForward或MoE结构
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        前向传播：
        1. 输入归一化后送入自注意力层，得到注意力输出和可选的past_key_value（用于缓存加速生成）。
        2. 残差连接：将注意力输出与原输入相加。
        3. 再归一化后送入MLP（或MoE），再与前一结果相加。
        4. 返回最终的hidden_states和present_key_value。
        Args:
            hidden_states: 输入隐藏状态 (batch, seq_len, hidden_size)
            position_embeddings: 位置编码（cos, sin）
            past_key_value: 历史缓存的key/value
            use_cache: 是否返回新的past_key_value
            attention_mask: 注意力掩码
        Returns:
            hidden_states: 输出隐藏状态
            present_key_value: 当前层的key/value缓存
        """
        residual = hidden_states  # 保存残差
        # 归一化后送入自注意力层
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual  # 残差连接
        # 再归一化后送入MLP（或MoE），再残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMindModel 是 MiniMind 的主干 Transformer 编码器模型。
    主要功能：
    - 负责将输入的 token id 序列编码为隐藏状态。
    - 支持多层 Transformer Block（可选普通/门控专家结构）。
    - 支持位置编码（RoPE），可缓存 past_key_values 用于高效自回归生成。
    - 输出最后一层隐藏状态、每层的 past_key_values 以及 MOE 辅助损失。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config  # 保存配置
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入层，将 token id 映射为向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)  # dropout 层，防止过拟合
        # 堆叠多个 Transformer Block，每层可为普通或 MOE 结构
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 输出层归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 位置编码（cos/sin），并注册为 buffer，避免每次前向都重复计算
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        前向传播：
        Args:
            input_ids: 输入 token id，形状(batch, seq_len)
            attention_mask: 注意力掩码
            past_key_values: 历史缓存的 key/value（用于自回归生成）
            use_cache: 是否返回新的 past_key_values
            **kwargs: 其他参数
        Returns:
            hidden_states: 最后一层隐藏状态 (batch, seq_len, hidden_size)
            presents: 每层的 past_key_values（用于生成）
            aux_loss: MOE 辅助损失（若有）
        """
        # eg：input_ids=tensor([[   1, 3169, 1951, 1794, 5819, 2885, 5392]])， batch_size=1 seq_length=7
        batch_size, seq_length = input_ids.shape
        # 如果没有 past_key_values，则初始化为 None 列表
        # eg: past_key_values = [None, None, None, None, None, None, None, None]
        past_key_values = past_key_values or [None] * len(self.layers)
        # 计算当前序列的起始位置（用于位置编码）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入 + dropout
        # eg：input_ids=Tensor(1,7,512)
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前位置的 RoPE 编码
        # eg：position_embeddings=(Tensor(7,64),Tensor(7,64))
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []  # 存储每层的 past_key_values
        # 依次通过每一层 Transformer Block
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最后一层归一化
        hidden_states = self.norm(hidden_states)

        # 累加所有 MOE 层的辅助损失（如果有）
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        # eg：hidden_sates=Tensor(1,7,512)，presents=[tuple( Tensor(1,7,2,64),Tensor(1,7,2,64) ) ...]  aux_loss=0
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind 因果语言模型类
    继承自 PreTrainedModel 和 GenerationMixin，用于实现自回归语言模型
    """
    config_class = MiniMindConfig  # 配置类

    def __init__(self, config: MiniMindConfig = None):
        """
        初始化 MiniMind 因果语言模型
        
        Args:
            config: MiniMindConfig 配置对象，如果为 None 则使用默认配置
        """
        self.config = config or MiniMindConfig()  # 使用传入的配置或默认配置
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)  # 创建 MiniMind 模型
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 语言模型头
        self.model.embed_tokens.weight = self.lm_head.weight  # 共享词嵌入和语言模型头的权重

        # CausalLMOutputWithPast 是Hugging Face Transformers库中的一个类，继承自ModelOutput基类，主要用于因果语言模型（Causal Language Model）的输出。
        self.OUT = CausalLMOutputWithPast()  # 输出容器

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        前向传播函数，由 GenerationMixin 类的generate方法调用
        
        Args:
            input_ids: 输入序列的 token IDs eg：tensor([[   1, 3169, 1951, 1794, 5819, 2885, 5392]])
            attention_mask: 注意力掩码 eg：tensor([[1, 1, 1, 1, 1, 1, 1]])
            past_key_values: 缓存的键值对，用于加速生成 eg：None
            use_cache: 是否使用缓存 eg：True
            logits_to_keep: 要保留的 logits 数量 eg：0
            **args: 其他参数 eg：{'inputs_embeds': None, 'return_dict': True}
            
        Returns:
            CausalLMOutputWithPast 对象，包含:
            - last_hidden_state: 最后一层的隐藏状态
            - logits: 预测的 logits
            - aux_loss: 辅助损失（如果使用 MOE）
            - past_key_values: 缓存的键值对
        """
        # 获取模型输出
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 计算要保留的 logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        

        # last_hidden_state 模型最后一层的隐藏状态，形状通常是(batch_size, sequence_length, hidden_size)
        # 包含了输入序列经过所有Transformer层处理后的最终表示，可以用于下游任务或进一步的特征提取
        self.OUT.__setitem__('last_hidden_state', h)
        # logits 模型输出的未归一化的概率分布，形状为(batch_size, sequence_length, vocab_size)
        # 表示每个位置对词汇表中每个词的概率预测，通过softmax可以转换为概率分布，用于预测下一个token
        self.OUT.__setitem__('logits', logits)
        # aux_loss: 辅助损失值，通常用于多任务学习或特殊训练目标，在MoE（Mixture of Experts）模型中可能表示专家路由的损失，帮助模型在主要任务之外优化其他目标   
        self.OUT.__setitem__('aux_loss', aux_loss)
        # past_key_values: 存储了注意力机制中的key和value状态
        # 用于自回归生成时的缓存机制，可以避免在生成每个新token时重新计算整个序列的注意力，显著提高生成效率 
        self.OUT.__setitem__('past_key_values', past_kvs)
        
        return self.OUT
