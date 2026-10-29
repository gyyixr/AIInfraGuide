---
title: "第2章：PPO 与价值学习"
description: "用概率比与优势解释 PPO 裁剪，区分参考 KL 和策略更新约束，再手算 token 奖励、TD 残差、GAE、Actor loss 与 Critic target"
pubDate: 2026-10-29
category: "llm-rl"
order: 20
tags: ["PPO", "GAE", "Critic", "KL", "价值学习"]
---

## 📖 本章概述

策略梯度告诉我们如何用奖励更新生成概率，但直接多次拟合同一批高分回答可能让策略变化过大。PPO 引入旧策略概率作为参照，并对代理目标中“已经足够改善”的方向裁剪收益。

同时，Critic 学习每个前缀的预计回报，GAE 把一步预测误差沿轨迹向前传递。前者关心**一次更新走多远**，后者关心**每个动作应该得到多少学习信号**。

本章使用一条三 token 回答贯穿奖励、优势和更新，使算法能落到具体数组，而不仅是一张四模型流程图。

## 📑 第一部分：PPO 的更新目标（正文 2.1）

### 1. old policy 与 current policy

对已经采样的 token，定义：

$$
\rho_t=\exp\big(\log\pi_\theta(y_t\mid s_t)-\log\pi_{\mathrm{old}}(y_t\mid s_t)\big)
$$

旧概率在本批训练期间保持不变；当前概率随 Actor 更新而改变。参考策略不承担这里的分母职责。

### 2. 按优势方向理解裁剪

PPO 最大化：

$$
\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)
$$

- 正优势：提高概率是有利方向，但超过上界不再继续奖励。
- 负优势：降低概率是有利方向，但低于下界不再继续奖励。
- 若走向有害方向，目标通常仍保留纠正梯度。

例如 $\epsilon=0.2,A=2,\rho=1.4$，代理收益取 2.4；若 $A=-2$，同样的 ratio 对应收益 -2.8，仍要惩罚概率增长。

### 3. KL、entropy 与 value loss

参考 KL 用于控制策略相对参考模型的偏移；old/current KL 用于监测单轮更新幅度。两者参照对象不同。Entropy 描述分布多样性，Critic loss 则训练价值估计；这些指标不能用一个“KL 正常”概括。

## 📑 第二部分：价值学习与完整迭代（正文 2.2）

### 4. TD 残差与 GAE

$$
\delta_t=r_t+\gamma V_{t+1}-V_t,\qquad
A_t=\delta_t+\gamma\lambda A_{t+1}
$$

正常终止后未来价值为 0。被采样长度上限截断的轨迹，是否 bootstrap 取决于任务定义，必须与 EOS 终止区分。

### 5. 三 token 算例

奖励为 $[-0.01,-0.02,0.97]$，旧价值为 $[0.2,0.4,0.6]$，$\gamma=1,\lambda=0.9$，则：

- TD 残差：$[0.19,0.18,0.37]$。
- GAE：$[0.6517,0.513,0.37]$。
- 价值回归目标：$[0.8517,0.913,0.97]$。

正文逐项计算这些值，并把它们带入 PPO policy loss 与 Critic loss。

### 6. 一轮训练的冻结边界

先采样并评分，再固定 old logprob、old value、advantage 和 value target；只在各 minibatch 中重新计算 current logprob 与 current value。完成有限次数更新后，发布下一版本权重并采样新数据。

## 🎯 自测与参考答案

1. PPO clip 是否保证所有 token 的 ratio 都在 $[0.8,1.2]$？
2. 为什么一个 batch 的 policy loss 为 0，Actor 仍可能有非零梯度？
3. 上述三 token 算例中，最后一个优势为什么是 $0.97-0.6$？
4. 为什么不能把归一化后的优势直接加到 value 上作为 Critic target？

**参考答案：** 1. 不保证，裁剪的是代理收益，参数共享也会改变被裁剪 token。2. 优势求和可为 0，但每项 logprob 对参数的导数不同。3. 末尾真正终止，未来价值是 0。4. Critic target 要处于原始回报尺度；优势标准化改变了尺度。

## 📚 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [InstructGPT](https://arxiv.org/abs/2203.02155)

这里展示的是可手算教学迭代，不是特定训练框架的性能或效果复现。
