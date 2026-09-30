# 📍 Day 4: 拼装完整 Transformer Decoder 与 MoE 架构

## 🎯 今日目标
1. [ ] 掌握前馈网络升级版：**SwiGLU（Swish Gated Linear Unit）**
   * 经典 FFN：$\text{ReLU}(x W_1 + b_1) W_2 + b_2$
   * SwiGLU：$(\text{SiLU}(x W_1) \odot (x W_3)) W_2$
   * 门控分支 $W_3$ 的物理含义：引入元素级的动态特征过滤，大幅提升非线性表达能力。
2. [ ] 残差连接（Residual Connection）的意义：
   * 为什么每个子层都要写成 `h = x + SubLayer(Norm(x))`？
   * 梯度高速通道：让反向传播梯度可以直接无衰减地穿透浅层，避免深度网络梯度消失。
3. [ ] 拼装完整架构：从 Token ID 到最终 Logits 的全链路：
   * `x = tok_embeddings(tokens)`
   * 经过 $N$ 个 `MiniMindBlock`（每层包含 Attention + FFN）
   * 最终 `norm(x)` 并送入 `output = lm_head(x)` 得到词表大小的预测分布。
4. [ ] （选学）MoE 混合专家模型机制：
   * `MoEGate`：利用一个小的线性层计算每个 token 分配给 4 个专家的概率。
   * Top-1 Routing：选取得分最高的专家进行 FFN 计算。
   * 为什么需要负载均衡辅助损失（Load Balancing Loss）？

## 🔍 涉及文件
* [model/model_minimind.py](file:///Users/kevin/Code/github/AI/minimind/model/model_minimind.py)
  * `FeedForward` 类（SwiGLU 实现）
  * `MiniMindBlock` 类（残差与 Pre-Norm 串联）
  * `MiniMindModel` 与 `MiniMindForCausalLM` 类
  * `MoEGate` 与 `MOEFeedForward` 类

## 🧪 今日实验指引
* `exp1_swiglu_vs_relu.py`：对比标准 MLP 与 SwiGLU 结构的参数量与非线性拟合能力。
* `exp2_forward_tensor_flow.py`：传入一组虚拟的 batch 数据，打印每一层输入输出的 Tensor Shape。
* `exp3_moe_routing_simulation.py`：模拟输入一组 token，观察 Top-1 路由将 token 分配给不同专家的分布。
