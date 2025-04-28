from transformers import PretrainedConfig
from typing import List


class LMConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dim: int = 512,
            n_layers: int = 8,
            n_heads: int = 8,
            n_kv_heads: int = 2,
            vocab_size: int = 6400,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 8192,
            rope_theta: int = 1e6,
            dropout: float = 0.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            ####################################################
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: bool = True,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs,
    ):
        # 基础模型参数
        self.dim = dim  # 模型隐藏层维度，决定模型容量和参数量
        self.n_layers = n_layers  # Transformer 层数，影响模型深度
        self.n_heads = n_heads  # 注意力头数，影响并行处理能力
        self.n_kv_heads = n_kv_heads  # Key-Value 注意力头数，用于分组注意力
        self.vocab_size = vocab_size  # 词表大小，决定模型可处理的词汇量
        self.hidden_dim = hidden_dim  # 前馈网络隐藏层维度，默认值为 4 * dim
        self.multiple_of = multiple_of  # 确保维度是此数的倍数，优化硬件性能
        self.norm_eps = norm_eps  # LayerNorm 的 epsilon 参数，防止除零
        self.max_seq_len = max_seq_len  # 最大序列长度，决定模型可处理的文本长度
        self.rope_theta = rope_theta  # RoPE 旋转位置编码的基础参数
        self.dropout = dropout  # Dropout 比率，用于防止过拟合
        self.flash_attn = flash_attn  # 是否使用 Flash Attention 优化
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        # MOE（混合专家）参数
        self.use_moe = use_moe  # 是否使用混合专家模型
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        super().__init__(**kwargs)
