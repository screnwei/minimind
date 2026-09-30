import sys
import os
import importlib.util

# 加载项目根目录的 model/model_minimind.py
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
spec = importlib.util.spec_from_file_location("model_minimind", os.path.join(root_dir, "model/model_minimind.py"))
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

# 直接暴露 MiniMind-3 的最新核心类
RMSNorm = _mod.RMSNorm
precompute_freqs_cis = _mod.precompute_freqs_cis
apply_rotary_pos_emb = _mod.apply_rotary_pos_emb
apply_rotary_emb = _mod.apply_rotary_pos_emb  # 兼容旧 notebook 命名
repeat_kv = _mod.repeat_kv
Attention = _mod.Attention
FeedForward = _mod.FeedForward
MiniMindBlock = _mod.MiniMindBlock
MiniMindModel = _mod.MiniMindModel
MiniMindForCausalLM = _mod.MiniMindForCausalLM
MiniMindConfig = _mod.MiniMindConfig

# 兼容旧 notebook 调用名称
MiniMindLM = MiniMindForCausalLM
