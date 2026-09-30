# 📓 MiniMind-3 交互式 Jupyter Notebook 学习沙盒

> 基于社区贡献者 [@Nijikadesu (PR #214)](https://github.com/jingyaogong/minimind/pulls/214) 的优秀交互式设计，全面升级适配至 **MiniMind-3（Qwen3 架构，64M/198M-MoE）** 规范。

---

## 🌟 为什么需要这些 Notebook？

* **告别“黑盒训练”**：传统学习方式是在终端直接跑脚本看 Loss 下降，而在这里，你可以单步执行每个 Cell，把 Tensor 打印出来看形状（Shape）和数值！
* **极速本地调试**：内置 `notebook/toydata/` 极简微型数据集（每个任务仅 2~3 条数据），任何普通个人电脑（包括 Mac CPU / MPS）**均可在几秒内完整跑完 1 次前向传播与反向传播**！
* **与主线代码 100% 对齐**：直接调用项目主干的 `model.model_minimind` 和 `dataset.lm_dataset`，学习到的每一行代码都能直接迁移到生产训练中。

---

## 🗺️ 笔记本与国庆 7 天学习计划联动映射表

| 笔记本文件 | 对应主题 | 核心拆解内容 | 联动计划 |
| :--- | :--- | :--- | :--- |
| **`0_tokenizer.ipynb`** | 分词器从零训练 | BPE 字节对编码算法、MiniMind-3 特殊标记（`<|im_start|>`, `<think>`, `<tool_call>`） | **Day 1** |
| **`1_model.ipynb`** | 模型结构模块化拆解 | RMSNorm、RoPE 旋转、GQA（8头与4头广播）、SwiGLU 门控前馈、完整 Decoder | **Day 2 ~ Day 4** |
| **`2_dataset.ipynb`** | 数据流与掩码探秘 | 预训练 Next-Token 偏移对齐、SFT Prompt Loss Masking（`-100` 梯度忽略） | **Day 3** |
| **`3_pretrain.ipynb`** | 预训练极简循环 | 在 2 条数据上完成 1 个 step 的 `forward -> loss -> backward -> step` | **Day 5** |
| **`4_sft.ipynb`** | 指令微调极简循环 | 对话格式渲染、Prompt 掩码过滤、单步全参数微调验证 | **Day 6** |
| **`5_dpo.ipynb`** | DPO 强化学习拆解 | 偏好样本对（Chosen vs Rejected）、隐式奖励差值计算与 DPO Loss 单步更新 | **Day 6 ~ Day 7** |
| **`6_lora.ipynb`** | 手写原生 LoRA 微调 | 低秩矩阵 $A \times B$ 线性层注入、主干权重冻结、极小参数量微调与权重合并 | **Day 6** |
| **`7_distill.ipynb`** | 知识蒸馏极简循环 | Teacher 模型 Logits 引导与 KL 散度损失计算 | **进阶选学** |
| **`8_reason.ipynb`** | 思考链与推理拆解 | `<think>...</think>` 思维链数据的输入构造与自适应思考机制 | **Day 7** |

---

## 🚀 快速开始运行

在激活的虚拟环境中启动 Jupyter Lab 或 Notebook：

```bash
# 激活环境
conda activate minimind

# 安装 jupyter（若未安装）
pip install jupyterlab ipywidgets

# 启动交互服务
jupyter lab
```

然后在浏览器或 VSCode 中打开 `notebook/` 下的任意 `.ipynb` 文件，即可逐单元格点击运行！
