---
title: "第10章：MoE 并行"
description: "从 Top-k 路由追踪 token 的 dispatch、专家计算与 combine，手算容量、负载和通信，再理解专家并行与其他并行维度的组合"
pubDate: 2026-04-16
updatedDate: 2026-10-30
category: "distributed-training"
order: 100
tags: ["MoE", "Expert Parallelism", "All-to-All", "负载均衡", "并行折叠"]
---

## 本章概述

稠密模型的 FFN 对每个 token 使用同一组权重。MoE 则准备多个专家，只让每个 token 进入其中少数几个。它扩大了可用参数容量，但引入了一项普通稠密训练没有的系统工作：根据每次变化的路由，把 token 送到持有目标专家的设备，再把结果送回来。

“激活参数较少”并不意味着其余参数可以完全不存，也不意味着通信成本按激活比例下降。真正的账本要同时计算专家权重、优化器状态、被复制的 token、路由元数据与最忙设备的执行时间。

本章用两个专题串起这条链，不把“调用一次 All-to-All”当作完整解释。

## 10.1 MoE 路由与专家并行

第一篇从一个六 token、四专家、两设备的例子出发。

| 环节 | 要回答的问题 | 可检查的产物 |
|---|---|---|
| Router | 每个 token 选哪些专家，权重如何归一化 | `[token, expert, gate]` 表 |
| 专家放置 | 哪台设备持有哪些专家 | 全局专家 ID 到设备的映射 |
| Dispatch | 本地排序、发送与接收数量怎样对应 | `send_splits` / `recv_splits` |
| Expert | 每个专家的矩阵乘法输入是多少行 | 每专家 token 计数与 shape |
| Combine | 如何恢复原 token 顺序并累加 Top-k 分支 | 六个最终输出值 |
| Backward | 输入和 gate 梯度分别从哪里回来 | 加权梯度与反向通信关系 |

读完应能手工复原整个例子，而不只是记住“前向两次 All-to-All”。

### 本节特别要区分的概念

- **EP** 分配不同专家；**ETP** 切分单个专家内部的矩阵。
- 同一专家的多个数据副本属于 **expert data parallel**，需要同步其梯度。
- token 的“逻辑专家分配条数”与实际跨网络发送次数不同，同一目标设备可能复用一份输入。
- capacity 是每专家可接收的配额；drop、padding 与 dropless 是不同处理策略。

## 10.2 MoE 负载均衡与通信优化

第二篇从同一组计数继续，分析训练中真正限制吞吐的因素。

| 主题 | 本篇重点 |
|---|---|
| 均衡指标 | 均值、最大值、CV、设备级聚合与尾延迟 |
| Router 目标 | 辅助损失、z-loss、动态 bias 各自改变什么 |
| 容量策略 | 为什么总槽位足够，某个专家仍可能溢出 |
| 计算效率 | 小 GEMM、Grouped GEMM、块稀疏与 padding 成本 |
| 通信效率 | payload、拓扑、split 元数据、permutation 与重叠 |
| 配置阅读 | 固定 Megatron 源码快照中的 dispatcher 和 group 边界 |

不会给出“EP 一定比 TP 快”这样的无条件结论。专家大小、token 数、Top-k、硬件拓扑与路由分布都会改变答案。

## 与前面章节怎样衔接

1. 第 2 章给出 All-to-All 的交换语义，本章补充不等长 split 与动态路由。
2. 第 5 章解释状态分片，本章需要分别统计稠密参数和专家参数的副本组。
3. 第 6 章解释 TP/SP，本章进一步说明专家 TP 可以不同于 Attention TP。
4. 第 7 章解释流水线，本章说明一个过载 MoE 层怎样拖慢整个 stage。
5. 第 11 章将专家布局放回完整 rank 网格，避免重复计算 world size。

## 建议的动手顺序

先运行 10.1 的标准库路由模拟，核对每个专家接收的 token 和 combine 输出。

再把某些 token 改为集中选择一个专家，观察最大负载、capacity drop 和设备级行数怎样变化。

最后才把这个逻辑映射到真实后端：固定版本，打印 expert ID、split sizes、输入输出 shape，并用小 batch 比较单设备参考实现与分布式结果。

本章的路由与成本计算可以在 CPU 上验证；它们不验证真实 NCCL/DeepEP 的吞吐，也不能代替多卡梯度等价性测试。

## 本章验收

- 能解释 Top-2 会产生多少条逻辑专家分配，以及何时能减少物理发送。
- 能手算一个不均衡 batch 的专家负载、设备负载和溢出条数。
- 能给出“专家均衡但设备仍不均衡”的反例。
- 能说明辅助损失优化目标与通信 kernel 优化目标的差别。
- 能区分 dense DP、EP、ETP、expert DP，并知道它们不能无条件全部相乘。
- 能把 MoE 一次前向拆成路由、排列、通信、GEMM、反向排列和归并六类开销。

## 自测提示

**只有一半专家被选中，是否只需要保存一半模型参数？** 一般不能。下一批路由可能使用其他专家；卸载或缓存是另一套明确设计，涉及加载延迟和状态管理。

**每个专家 token 数相同，是否说明训练已最佳？** 不一定。专家矩阵大小、设备放置、跨节点流量、token padding 和流水线阶段耗时都可能不均衡。

**“Aux-loss-free”是否意味着训练中绝无任何辅助项？** 不一定。某些方法用 bias 调节替代主要均衡损失，同时仍保留其他辅助目标，必须阅读完整配方。

## 参考资料

- [Switch Transformers](https://arxiv.org/abs/2101.03961)：稀疏路由、专家容量与均衡损失。
- [MegaBlocks](https://arxiv.org/abs/2211.15841)：动态专家计算与 dropless 实现。
- [MoE Parallel Folding](https://arxiv.org/abs/2504.14960)：Attention 与专家网格的不同映射。
- [Megatron Core MoE 版本文档](https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/user-guide/features/moe.html)：配合专题中固定源码阅读，不将滚动默认值视为算法定义。
