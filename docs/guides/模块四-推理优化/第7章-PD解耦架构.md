---
title: "第7章：Prefill/Decode 解耦架构"
description: "从 P/D 干扰、服务需求、KV 传输与异步取消建立可验证的解耦系统"
pubDate: 2026-04-16
updatedDate: 2026-11-07
category: "inference-optimization"
chapter: 7
order: 70
tags: ["PD解耦", "KVConnector", "HiCache", "Goodput", "生命周期"]
---

## 本章要解决的问题

P/D 解耦允许分别配置 prefill 和 decode 的硬件、并行和调度。
它可能减少长 prefill 对逐 token 延迟的干扰。
代价是 KV 交接、双池排队、重复驻留和更复杂的故障处理。
因此是否解耦应由 SLO 与测量决定。

本章不使用没有配置条件的“必然提速若干倍”结论。
Prefill/decode 的 compute/memory bound 也只是待验证的起点。

## 学习路径

| 小节 | 核心问题 | 产出 |
|---|---|---|
| [7.1 P/D 解耦的成本模型与调度](./7.1-PD解耦的成本模型与调度.md) | 何时值得拆池 | KV 字节、P:D 配比、实验矩阵 |
| [7.2 KV Connector 与 HiCache 分层缓存](./7.2-KVConnector与HiCache分层缓存.md) | 不同 KV 层各管什么 | owner 与恢复成本模型 |
| [7.3 KV 传输、取消、重试与生命周期](./7.3-KV传输取消重试与生命周期.md) | 取消后何时能 free | generation、drain 与幂等状态机 |

先完成第 6 章的资源记账。
再把一个请求从 gateway 一直追到最后一次异步写入结束。

## 1. 先有 aggregated 与 chunked 基线

最小基线通常是共享池的 continuous batching。
长 prefill 造成干扰时，先测 chunked prefill 和 token budget。
然后比较按 workload 分池，最后评估完整 PD。

```text
aggregated
  → chunked prefill
  → workload isolation
  → P/D disaggregation
```

这是实验顺序，不是必须逐级采用的技术路线。
若已有证据证明双池硬件收益，可以直接验证 PD 候选。
所有对照应固定总 GPU 预算、模型和请求分布。

## 2. 一次请求的状态路径

```text
admission
  → D reservation
  → P queue and prefill
  → KV transfer
  → D commit
  → decode
  → stream completion
  → async retirement
```

具体实现可能先执行 P 再选择 D。
两种顺序对目标预分配、缓存和背压有不同要求。
必须画出实际协议，不能把两张 GPU 图当完整设计。

## 3. 服务需求与逗留时间

配比公式需要的是平均副本秒服务需求。
Little's Law 使用的是请求平均逗留时间。
Continuous batching 会让多个请求共享一个 decode step。
不能把每条请求的 wall time 都当作独占副本时间。

$$
n_P\ge\lceil\lambda d_P/u_P\rceil,
\qquad n_D\ge\lceil\lambda d_D/u_D\rceil.
$$

算完副本数，还要乘各阶段每副本 GPU 数。
P:D 副本比不一定等于 P:D GPU 比。
最后用真实 trace 修正 batch 效率、burst 与故障余量。

## 4. 五层 KV 系统

| 层 | 职责 |
|---|---|
| Allocator | 本地页面与 block table |
| Prefix cache | 连续语义前缀复用 |
| Hierarchical cache | GPU/CPU/存储驻留 |
| P→D handoff | 单请求阶段交接 |
| Router hint | 选择副本的预测信息 |

Connector 可以连接其中若干层，但不让这些职责自动合并。
Router 只持提示，worker 必须实际校验缓存。
缓存 key 还需要模型、adapter、位置和安全域身份。

## 5. 两种不同的 break-even

PD 比较的是阶段隔离收益与额外交接、排队、碎片成本。
Remote prefix/offload 比较的是恢复 KV 与重新计算 miss 的成本。
二者不能使用同一个“省了 prefill”理由。

KV 传输至少包含固定协调、payload、排队和 layout 转换。
Layer-wise pipeline 可以隐藏部分时间，但引入逐层 ready 与失败状态。
不同 stream 上有活动，不等于关键路径真的缩短。

## 6. 取消后不能立即复用页面

```text
FREE → RESERVED → FILLING → COMMITTED → RETIRING → FREE
```

HTTP stream 结束，不代表 kernel 或 DMA 已结束。
请求 generation 可以拦旧控制消息，但不自动阻止裸远端 write。
回收需要覆盖所有 writer 的 drain/completion 或更强的数据面隔离。
Timeout 是恢复策略的一部分，不是安全证明。

## 7. 本章动手任务

1. 算 8K/32K GQA prompt 的 KV 字节和传输下界。
2. 依据给定副本秒需求算 P:D 配比与 GPU 数。
3. 比较本地恢复、远端恢复和重算时间。
4. 模拟重复 ack、旧 generation 和未完成本地事件。
5. 检查取消后页面总量与 owner 是否守恒。

这些小实验只依赖 Python 标准库。
它们不会启动真实 RDMA，也不代表多机性能验证。

## 8. 学完后应该能回答

- PD 为什么可能提高 goodput 却降低 raw throughput？
- 首 token 由 P 返回后，handoff 还会影响哪个指标？
- P 与 D 的 TP 不同需要哪些布局条件？
- Prefix hit 为何不等于 GPU 页面立即可读？
- Refcount 为零为何仍可能发生 UAF？
- 传输 timeout 后怎样同时满足安全与活性？

每个答案应指出 owner、时间边界和失败条件。

## 9. 来源与范围

参考宝典映射 `KV-008—020`、`SCH-012/013/018/019`、`DIST-016—020`。
细节同时关联 `CODE-005/019`、`PROD-014` 与 `DESIGN-003/005/011`。

[SGLang 固定 decode 源码](https://github.com/sgl-project/sglang/blob/46d9427b913b20ce0d72b95a9209717448b37368/python/sglang/srt/disaggregation/decode.py) 提供延迟回收的静态证据。
[vLLM 固定 connector 源码](https://github.com/vllm-project/vllm/blob/9c8e90eb2637a863ca14e47fd436b10ed7ba6536/vllm/distributed/kv_transfer/kv_connector/v1/base.py) 提供生命周期契约。

这些引用不表示执行过多节点部署。
继续阅读第 8 章，用统一指标比较聚合与解耦方案。
