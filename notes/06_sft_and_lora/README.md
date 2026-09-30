# 📍 Day 6: 监督微调（SFT）与手写原生 LoRA——让模型听懂人话

## 🎯 今日目标
1. [ ] 理解预训练（Pretrain）到监督微调（SFT）的质变：
   * 预训练让模型拥有“语言能力与世界常识”，SFT 让模型建立“问答助手角色与指令服从”。
2. [ ] 攻克 SFT 的核心技术基石：**Prompt Loss Masking（提示词掩码机制）**
   * 为什么绝不能对用户的提问计算 Loss？如果计算了会有什么灾难性后果？
   * PyTorch 中的 `CrossEntropyLoss(ignore_index=-100)`：如何把用户问句部分对应的 label 全部填充为 `-100`，只对 Assistant 的回答计算损失。
3. [ ] 手写原生 LoRA（Low-Rank Adaptation）从零实现：
   * 原理：$W_{new} = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \times A)$
   * 为什么矩阵 $A \in \mathbb{R}^{r \times d_{in}}$ 用高斯随机初始化，而矩阵 $B \in \mathbb{R}^{d_{out} \times r}$ 必须初始化为全 0？（确保初始状态下 $\Delta W = 0$）
   * 为什么设置 `param.requires_grad = False` 冻结主干权重后，只需训练不足 1% 的参数？
4. [ ] 权重合并与导出：
   * 训练完成后，如何直接将 $\frac{\alpha}{r}(B \times A)$ 加回到原权重 $W_0$，实现零额外推理开销。

## 🔍 涉及文件
* [dataset/lm_dataset.py](file:///Users/kevin/Code/github/AI/minimind/dataset/lm_dataset.py)（`SFTDataset` 类中的掩码拼接实现）
* [trainer/train_full_sft.py](file:///Users/kevin/Code/github/AI/minimind/trainer/train_full_sft.py)（全参数微调流程）
* [model/model_lora.py](file:///Users/kevin/Code/github/AI/minimind/model/model_lora.py)（原生手写 LoRA 线性层）
* [trainer/train_lora.py](file:///Users/kevin/Code/github/AI/minimind/trainer/train_lora.py)（LoRA 微调脚本）
* [scripts/convert_model.py](file:///Users/kevin/Code/github/AI/minimind/scripts/convert_model.py)（权重合并脚本）

## 🧪 今日实验指引
* `exp1_prompt_masking_demo.py`：打印一条多轮对话数据，用彩色终端输出哪些 token 参与了 Loss 计算，哪些 token 的 label 为 `-100`。
* `exp2_lora_math_and_forward.py`：手动写一个包含 LoRA 分支的 Linear 层，验证在初始状态下输出与原始层完全相同，并在 backward 时只有 A 和 B 获得梯度。
* `exp3_mini_lora_train.py`：利用 `dataset/lora_identity.jsonl` 做极微量训练，验证让模型回答特定专属身份。
