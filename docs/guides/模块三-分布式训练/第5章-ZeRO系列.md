---
title: "第5章：ZeRO 显存优化与 FSDP2"
description: "从状态所有权推导 ZeRO-1/2/3 的显存和通信账本，进一步理解 FSDP2 的参数生命周期、预取、Offload 与分布式保存"
pubDate: 2026-04-16
updatedDate: 2026-10-30
category: "distributed-training"
order: 50
tags: ["ZeRO", "FSDP2", "DTensor", "显存优化", "Offload"]
---

## 本章要解决的问题

数据并行通过复制模型分担数据，但每个 rank 重复保留优化器、梯度与参数。模型变大后，这些长期状态会先于计算能力达到显存上限。

本章把“计算用什么”和“平时由谁保存”分开。ZeRO 让各 rank 对状态分片负责；需要完整参数的计算发生前，再临时取得对应数据。

这和 TP 的区别很重要：ZeRO/FSDP 通常仍在每个数据 rank 上执行完整层计算；TP 则把同一个层内算子拆给多个 rank 合作执行。

## 阅读路线

| 专题 | 核心问题 | 读完应能完成 |
|---|---|---|
| [5.1 ZeRO 分片与显存通信账本](/AIInfraGuide/distributed/模块三-分布式训练/51-zero分片与显存通信账本/) | 每份状态由谁拥有、通信数据是什么 | 手算四参数更新和 7B/8 rank 账本 |
| [5.2 FSDP2 与 Offload 工程实践](/AIInfraGuide/distributed/模块三-分布式训练/52-fsdp2与offload工程实践/) | 分片怎样映射到参数组和内存生命周期 | 解释 fully_shard、预取、CPU/NVMe 代价 |

建议先完成第 2 章 ReduceScatter/AllGather 和第 3 章优化器状态。已有 FSDP1 使用经验，也应先理解 FSDP2 的 DTensor 参数表示，再迁移代码。

## 1. 三阶段看的是状态所有权

设计算参数、梯度、优化器相关状态总字节分别为 $P,G,O$，数据分片组大小为 $D$：

| 策略 | 长期分片状态 | 每 rank 理想静态量 |
|---|---|---|
| DDP | 无 | $P+G+O$ |
| ZeRO-1 | 优化器相关状态 | $P+G+O/D$ |
| ZeRO-2 | 再加梯度 | $P+(G+O)/D$ |
| ZeRO-3 | 再加计算参数 | $(P+G+O)/D$ |

这里的分母是对应状态分片组的大小。混合 TP/PP 时，应先确定该模型分片拥有多少参数，不能盲目除以集群总卡数。

## 2. 一套可复查的数值账本

采用 BF16 计算参数 2 Bytes、BF16 保留梯度 2 Bytes、FP32 主权重与两个 Adam moment 共 12 Bytes 的教学假设：

```text
7B 参数 × 16 Bytes = 112 GB
8 rank 的 ZeRO-1 = 38.5 GB/rank
8 rank 的 ZeRO-2 = 26.25 GB/rank
8 rank 的 ZeRO-3 = 14 GB/rank
```

这些数字是静态下限，不包含激活和聚合 buffer。真实框架若使用 FP32 梯度、没有独立主副本或采用其他优化器，必须重新计账。

FSDP2 的 FP32 shard 配合临时 BF16 完整参数是另一种生命周期表达，不能把每个对象同时按长期状态累加。

## 3. 通信不由 stage 编号唯一决定

每段梯度归约到拥有者，更新后同步参数，是理解分片优化器的一条清晰路径。

ZeRO-1 并非必然执行“完整梯度 AllReduce，然后额外完整参数 AllGather”。后端可以用拥有者归约、融合 bucket 等方式组织数据移动。

若每 rank 发送量采用 ring 模型，令 $c=(D-1)/D$：

- 梯度 AllReduce：$2cG$。
- 梯度 ReduceScatter：$cG$。
- 参数 AllGather：$cP$。

“ZeRO-3 比 DDP 多 50% 通信”依赖参数与梯度字节宽度相同、前后向各聚合一次等假设。它不是对所有混合精度实现的性能承诺。

## 4. 静态下限之外还要看活动峰值

实际峰值还包含当前参数组、下一组预取、激活、通信 buffer、算子 workspace 和分配器保留空间。

因此，选择 block 级还是整模型级参数组，会直接影响能否训练。只看到长期 shard 很小，不能证明前向前的 AllGather 一定装得下。

应在加载、前向、反向、更新和保存阶段分别测峰值，不要只记录训练结束后的空闲显存。

## 5. Offload 的代价移到了哪里

CPU offload 可以搬走优化器状态及更新计算，参数 offload 则有自己的开关。主要前向和反向矩阵计算通常仍在 GPU。

代价包括 CPU RAM、锁页内存、PCIe 传输和 CPU optimizer 时间；NVMe 进一步扩展容量，但需要为读写带宽和预取窗口付费。

先估算每步搬动的字节，再用实测有效带宽计算下界，最后看哪些部分能被 GPU 计算隐藏。

## 动手与验收

1. 用四参数、两个 rank 的例子，写出 ReduceScatter 输出、拥有者更新和新参数聚合。
2. 修改梯度 dtype，重算 ZeRO 的显存和通信比例。
3. 在 FSDP2 示范中标注 optimizer 创建前后的参数对象。
4. 为一个 0.8 GB 参数组设计预取深度，并列出峰值的组成。
5. 给出 CPU offload 的单向传输下界，区分串行与双向重叠。

第 5.1 节含无需 GPU 的账本计算。第 5.2 节提供 GPU 训练示范，但未把静态核对或 CPU 实验冒充真实多 GPU FSDP2 验证。

## 自测与答案

**ZeRO-3 的参数 shard 很小，为什么仍会 OOM？** 当前完整参数组、激活与临时 workspace 可能一起超预算。

**为什么不能只比较两种方案的 NCCL 总时间？** 通信可能与计算重叠；真正影响 step 的是关键路径上暴露的等待。

**CPU offload 是否意味着反向也全部在 CPU？** 不是；应明确移动的是状态、optimizer 更新还是其他特定计算。

## 参考资料

- [ZeRO](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed ZeRO 配置](https://deepspeed.readthedocs.io/en/latest/zero3.html)
