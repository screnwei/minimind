import sys
import os
import importlib.util

# 加载项目根目录的 model/model_minimind.py
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
spec = importlib.util.spec_from_file_location("model_minimind", os.path.join(root_dir, "model/model_minimind.py"))
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

MiniMindConfig = _mod.MiniMindConfig

class LMConfig(MiniMindConfig):
    """
    MiniMind-3 交互式 Notebook 专用配置
    继承自最新的 MiniMindConfig (Qwen3 架构生态)
    默认设置 num_hidden_layers=1 便于在本地单机毫秒级调试
    """
    def __init__(
        self,
        hidden_size: int = 512,  # 笔记本教学设置 512 维，轻量快速
        num_hidden_layers: int = 1,  # 设为 1 层便于本地单机调试
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4, # MiniMind-3 GQA 升级为 4 个 KV 头
        vocab_size: int = 6400,
        intermediate_size: int = 2432, # SwiGLU 维度
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        use_moe: bool = False,
        **kwargs
    ):
        # 兼容旧 notebook 中可能传入的旧参数名
        if "dim" in kwargs:
            hidden_size = kwargs.pop("dim")
        if "n_layers" in kwargs:
            num_hidden_layers = kwargs.pop("n_layers")
        if "n_heads" in kwargs:
            num_attention_heads = kwargs.pop("n_heads")
        if "n_kv_heads" in kwargs:
            num_key_value_heads = kwargs.pop("n_kv_heads")

        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            use_moe=use_moe,
            **kwargs
        )

        # 挂载属性兼容旧代码访问
        self.dim = self.hidden_size
        self.n_layers = self.num_hidden_layers
        self.n_heads = self.num_attention_heads
        self.n_kv_heads = self.num_key_value_heads
        self.max_seq_len = self.max_position_embeddings
        self.hidden_dim = self.intermediate_size
        self.multiple_of = 64
        self.dropout = 0.0
        self.flash_attn = False
        self.norm_eps = getattr(self, 'rms_norm_eps', 1e-6)
        self.num_experts_per_tok = getattr(self, 'num_experts_per_tok', 1)
        self.n_routed_experts = getattr(self, 'n_routed_experts', 4)
        self.n_shared_experts = False
        self.aux_loss_alpha = 0.1
        self.seq_aux = True
        self.norm_topk_prob = True
        self.scoring_func = 'softmax'


