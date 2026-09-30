import sys
import os
import importlib.util

# 加载项目根目录的 dataset/lm_dataset.py
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
spec = importlib.util.spec_from_file_location("lm_dataset", os.path.join(root_dir, "dataset/lm_dataset.py"))
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

PretrainDataset = _mod.PretrainDataset
SFTDataset = _mod.SFTDataset
DPODataset = _mod.DPODataset
