# 📍 Day 2: 坐标与稳定器——RoPE 旋转位置编码与 RMSNorm

## 🎯 今日目标
1. [ ] 理解位置编码的本质：为什么 Attention 本身是置换不变的（Permutation Invariant）？为什么必须注入位置信息？
2. [ ] 攻克 RoPE（Rotary Position Embedding）的核心思想：
   * 为什么旋转二维向量可以天然编码相对距离？
   * 旋转矩阵与复数乘法的等价性。
   * 维度与频率衰减：为什么高维旋转慢（捕捉长程），低维旋转快（捕捉短程）？
3. [ ] 逐行搞懂代码实现：
   * `precompute_freqs_cis(dim, end, rope_base)`：计算基础频率 $\theta_i$ 与 cos/sin 网格。
   * `apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)`：把 Q 和 K 沿 head_dim 进行成对旋转。
4. [ ] 掌握稳定器 RMSNorm（Root Mean Square Normalization）：
   * 为什么相比 LayerNorm 丢掉了均值（Mean Centering）却不影响效果？
   * Pre-Norm 与 Post-Norm：为什么现代 LLM 全面转向 Pre-Norm？

## 🔍 涉及文件
* [model/model_minimind.py](file:///Users/kevin/Code/github/AI/minimind/model/model_minimind.py)
  * `RMSNorm` 类
  * `precompute_freqs_cis` 函数
  * `apply_rotary_emb` 函数

## 🧪 今日实验指引
* `exp1_why_position_needed.py`：测试同一个句子打乱词序，在没有位置编码时 Attention 分数矩阵是否完全相同。
* `exp2_rope_rotation_visualization.py`：构建一个 2D 向量，用 `apply_rotary_emb` 观察随着 position 增加，向量如何在平面上步进旋转。
* `exp3_rmsnorm_vs_layernorm.py`：对比 RMSNorm 和 LayerNorm 的计算公式、参数量与耗时。
