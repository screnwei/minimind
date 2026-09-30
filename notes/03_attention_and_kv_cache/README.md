# 📍 Day 3: 核心计算引擎——自注意力（Self-Attention）、GQA 与 KV Cache

## 🎯 今日目标
1. [ ] 从零推导自注意力机制公式：
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{Mask}\right) V$$
   * 搞懂 Q、K、V 的角色：Query 是“我在找什么”，Key 是“我是什么特征”，Value 是“我的实际内容”。
   * 为什么除以 $\sqrt{d_k}$？防止点积过大导致 Softmax 梯度饱和/消失。
2. [ ] 因果掩码（Causal Mask）：
   * 为什么自回归模型必须用下三角掩码？
   * 上三角被置为 $-\infty$（在 Softmax 后变为 0）的物理意义。
3. [ ] 攻克 GQA（Grouped-Query Attention）：
   * 对比 MHA（多头注意力）、MQA（多查询注意力）与 GQA。
   * MiniMind-3 设置：`q_heads=8, kv_heads=4`。
   * `repeat_kv` 函数：如何将 4 个 KV head 在维度上扩展到 8 个 head 与 Q 对齐。
4. [ ] 自回归推理加速器 KV Cache：
   * Prefill（首字）阶段：输入全长 prompt，计算并缓存每层的 Key 和 Value。
   * Decode（逐字）阶段：每次只输入 1 个新生成的 token，取出历史 KV 拼接，为什么复杂度从 $O(N^2)$ 降低到 $O(N)$？

## 🔍 涉及文件
* [model/model_minimind.py](file:///Users/kevin/Code/github/AI/minimind/model/model_minimind.py)
  * `repeat_kv` 函数
  * `Attention` 类（前向传播与缓存拼接逻辑）

## 🧪 今日实验指引
* `exp1_attention_matrix_walkthrough.py`：纯手写一个小型 Attention 计算，打印出权重热力图与掩码效果。
* `exp2_gqa_dimension_trace.py`：追踪 `q_heads=8` 与 `kv_heads=4` 在 `repeat_kv` 后的张量维度变化。
* `exp3_kv_cache_speedup.py`：模拟自回归生成，对比开启与不开启 KV Cache 时的推理延迟差异。
