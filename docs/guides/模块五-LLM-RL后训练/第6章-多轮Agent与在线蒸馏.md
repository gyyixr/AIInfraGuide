---
title: "第6章：多轮 Agent 与在线蒸馏"
description: "从单轮回答扩展到工具交互轨迹与教师反馈，理解动作 mask、终止语义、异步环境和 On-Policy 蒸馏的数据与梯度边界"
pubDate: 2026-10-29
category: "llm-rl"
order: 60
tags: ["Agent RL", "工具调用", "OPD", "知识蒸馏"]
---

## 📖 本章概述

单轮训练中，模型连续生成回答，再获得一个分数。多轮 Agent 则在生成之间插入工具结果、环境变化和新的决策；在线蒸馏在学生实际访问的前缀上引入教师反馈。这两类扩展都要求精确回答同一个问题：**谁生成了哪些 token，谁在什么条件下给它们打分？**

本章先建立多轮轨迹契约，再解释教师如何在学生轨迹上提供训练信号。建议已经掌握策略梯度、回答 mask 和参考模型的角色。

## 📑 章节结构

### 6.1 多轮轨迹与工具环境

一条轨迹可以写成：

```text
用户问题 → assistant 工具调用 → tool 观测 → assistant 继续推理
→ tool 观测 → assistant 最终回答 → 环境奖励
```

工具观测参与后续 attention，却不是 Actor 采样的动作。若把工具返回的 1,000 token 也放进 policy loss，优化器会尝试学习“生成工具结果”，而不是正确选择工具。

本节讲解：

- attention mask、action mask 和 padding mask 的不同含义。
- token、轮次和环境决策三个时间尺度。
- 最终奖励如何通过回报或优势分配到动作。
- 任务真正结束与外部截断对 bootstrap 的影响。
- 超时、重试、幂等、版本与异步调度的最小契约。

### 6.2 On-Policy 蒸馏与教师反馈

在线蒸馏首先定义的是**训练轨迹来自学生自己的策略**。教师可以在这些轨迹上提供全词表分布、Top-k 分布或实际 token 的 logprob；这些接口对应的计算量与目标并不相同。

本节用三词词表手算：

$$
p=(0.6,0.3,0.1),\qquad q=(0.2,0.5,0.3),
$$

其中 $p$ 为学生，$q$ 为教师。通过 Forward KL、Reverse KL 及 sampled-token 梯度对比，理解“同一轨迹”不等于“同一个 loss”。

尤其区分：

- 教师生成一个新回答，作为学生 SFT 标签。
- 教师读取学生回答，在每个学生前缀上提供概率。
- 固定前缀上的局部 KL 梯度。
- 完整序列分布目标的策略梯度与回报传播。

## 🧭 系统边界

| 对象 | 数据来源 | 是否进入 Actor 动作 loss |
|---|---|---|
| 用户/system 输入 | 数据集或环境 | 否 |
| assistant 生成 token | 学生/Actor 采样 | 通常是 |
| tool 返回文本 | 外部环境 | 否，但可作为后续条件 |
| 教师概率 | 冻结教师前向 | 作为监督信号，不更新教师 |
| 原始标准答案 | 数据集 | 仅按任务定义交给 verifier/教师 |

## 🛠️ 实践路径

先在一条人工构造轨迹上逐 token 标记角色和 mask，再尝试多轨迹并发。蒸馏先用三词词表验证 loss 与梯度，再接入真实模型和批量教师打分。

本章的算例验证数学与数据契约；不构成真实多工具环境、生产教师服务或多 GPU 蒸馏吞吐的测试结论。

## 🎯 自测与参考答案

**问题 1：tool token 的 action mask 为 0，是否应该从上下文中删除？**

不应。它仍然是后续决策的重要观测，通常 attention mask 为 1。

**问题 2：教师必须重新生成一段更好的回答，才能做 OPD 吗？**

不必须。它可以对学生已经生成的同一序列提供条件概率，这正是本章重点。

**问题 3：OPD 是否必然采用 PPO 或 Reverse KL？**

不是。轨迹来源、教师接口、散度和优化方式是不同选择，应逐项核对。

## 📚 阅读依据

- [verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)：多轮 rollout 的接口设计。
- [Gymnasium Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)：终止与截断的学习语义。
- [On-Policy Distillation / GKD](https://arxiv.org/abs/2306.13649)：学生轨迹上的教师反馈。
