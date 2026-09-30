# 📍 Day 1: 语言模型的起点——分词（Tokenizer）与词嵌入（Embedding）

## 🎯 今日目标
1. [ ] 彻底搞懂：文本是怎么从一串字符变成计算机能算矩阵的数字的？
2. [ ] 学习 Byte-level BPE 原理：既能保留完整中文语义，又能应对生僻字与多语言。
3. [ ] 掌握特殊标记（Special Tokens）的作用：
   * `<|im_start|>` 和 `<|im_end|>`（对话角色的分界线）
   * `<think>` 和 `</think>`（思维链标记）
   * `<tool_call>` 和 `</tool_call>`（工具调用标记）
4. [ ] 探索 `nn.Embedding` 查表机制：Token ID (整数) 是如何被投影到 768 维向量的。

## 🔍 涉及文件
* `model/tokenizer.json`（分词器词表定义）
* `model/tokenizer_config.json`（特殊 token 配置）
* `trainer/train_tokenizer.py`（从零训练分词器的脚本）
* `model/model_minimind.py`（`MiniMindModel.tok_embeddings` 嵌入层定义）

## 🧪 今日实验指引
* `exp1_tokenizer_basics.py`：测试常用中文句子、中英混合句子被 tokenize 后的 Token IDs 及解码还原。
* `exp2_embedding_lookup.py`：手动构建一个 `nn.Embedding`，观察从 token 序列查表得到 `[batch_size, seq_len, 768]` 浮点张量的完整过程。
