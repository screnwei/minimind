# 📍 Day 7: 偏好对齐（DPO）与前沿 Agent / 思考链初探

## 🎯 今日目标
1. [ ] 为什么 SFT 之后还需要强化学习偏好对齐（RLHF）？
   * 区分“能回答问题”与“回答得更好、更安全、更诚实”。
2. [ ] 攻克 DPO（Direct Preference Optimization，直接偏好优化）：
   * 传统 PPO：需要训练一个独立的 Reward Model（奖励模型），流程繁琐且极不稳定。
   * DPO 突破：利用数学变换证明了策略本身就能充当隐式奖励，直接通过好回答（chosen）与坏回答（rejected）的概率比优化目标：
     $$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$
   * 剖析 `train_dpo.py` 中的对数概率提取与 DPO 损失手写实现。
3. [ ] MiniMind-3 前沿特性探索：
   * **思考链（Reasoning）**：模型如何自主生成 `<think> ... </think>`，`open_thinking` 参数是如何通过 chat_template 引导思考开关的。
   * **工具调用（Tool Calling）与 Agentic RL**：
     * `<tool_call>` 的 JSON 格式协议。
     * 多轮对话状态机：User $\to$ Model 提出调用 $\to$ Tool 执行 $\to$ 结果回填 $\to$ Model 最终回答。
     * 原生智能体强化学习脚本 `train_agent.py` 与解耦的 `rollout_engine.py` 架构初窥。
4. [ ] 假期结课总结与个人 LLM 知识树梳理。

## 🔍 涉及文件与专题资料
* [trainer/train_dpo.py](file:///Users/kevin/Code/github/AI/minimind/trainer/train_dpo.py)（DPO 原生手写实现）
* [scripts/eval_toolcall.py](file:///Users/kevin/Code/github/AI/minimind/scripts/eval_toolcall.py)（工具调用评测与执行闭环）
* [trainer/train_agent.py](file:///Users/kevin/Code/github/AI/minimind/trainer/train_agent.py)（原生智能体强化学习）
* [trainer/rollout_engine.py](file:///Users/kevin/Code/github/AI/minimind/trainer/rollout_engine.py)（采样引擎）
* 📖 **深度专题精读**：[deep_dive_web_search_and_agent.md](file:///Users/kevin/Code/github/AI/minimind/notes/07_dpo_and_agent/deep_dive_web_search_and_agent.md)（大模型 API 自带“联网搜索”是如何实现的？从原理、架构时序到 MiniMind 代码实战）

## 🧪 今日实验指引
* `exp1_dpo_loss_walkthrough.py`：单步模拟 DPO 损失计算，输入一个好回答与坏回答，观察随着模型给好回答赋予更高概率时，Loss 如何下降。
* `exp2_toolcall_runner.py`：运行 `eval_toolcall.py`，观察模型调用计算器、天气等函数的完整交互协议与输出。
