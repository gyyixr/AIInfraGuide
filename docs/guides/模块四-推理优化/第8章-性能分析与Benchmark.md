---
title: "第8章：性能分析与 Benchmark"
description: "统一指标和负载口径，从端到端 Trace 定位热点，并用质量、SLO 和成本联合门禁验证收益"
pubDate: 2026-04-16
updatedDate: 2026-11-07
category: "inference-optimization"
chapter: 8
order: 80
tags: ["Benchmark", "Profiling", "TTFT", "TPOT", "Goodput", "性能回归"]
---

## 本章要解决的问题

推理性能不仅是 kernel 执行时间。
用户等待还包含认证、tokenization、排队、KV 恢复和网络写出。
吞吐增加也可能以更差的 tail、质量或错误率为代价。
本章建立一套能重算、能解释、能回归的测量方法。

本章小节编号统一为 8.x。
原导读页面保留，正文扩展为三篇详细教程。

## 学习路径

| 小节 | 核心问题 | 产出 |
|---|---|---|
| [8.1 推理 Benchmark 指标与负载模型](./8.1-推理Benchmark指标与负载模型.md) | 测什么、怎样造负载 | 时间戳、指标、open/closed-loop |
| [8.2 从端到端 Trace 到 GPU Kernel 定位](./8.2-从端到端Trace到GPUKernel定位.md) | 慢在哪里、为什么慢 | 关键路径、Nsys/NCU、收益上限 |
| [8.3 性能回归、质量护栏与成本优化](./8.3-性能回归质量护栏与成本优化.md) | 优化是否值得上线 | 联合门禁、交错 A/B、灰度回滚 |

先理解第 1—2 章的执行阶段和调度。
无需 GPU 也可以运行本章时间戳、Roofline 与门禁小算例。
真实硬件结论需要另做受控采集。

## 1. 指标从原始事件计算

```text
scheduled arrival
→ actual send
→ first visible token
→ later visible tokens
→ last visible token
→ final stream completion
```

TTFT 从明确的发送/到达边界算到首 token。
ITL 是相邻 token 的间隔。
TPOT 是首 token 后的请求内平均，不能代替 ITL 分布。
E2E 算到请求最终完成。

单 token 请求的 TPOT 未定义，不应填 0。
SSE chunk 也不一定等于一个 token。
客户端与服务端时间边界必须分别标注。

## 2. Goodput 与吞吐

Output TPS 统计完成的输出工作量。
Goodput 只统计满足约定成功、延迟与质量条件的交付。
一个配置可能 raw TPS 更高，但 goodput 更低。

$$
Goodput=\frac{\text{满足联合要求的完成请求数}}{\text{测量时长}}.
$$

分母还需解释是否包含冷启动、排空和跨窗口请求。
拒绝、失败、取消和超时不能在报告里消失。
关键结果至少包含 p50/p95/p99、样本数和重复运行波动。

## 3. Open-loop 与 closed-loop

Open-loop 按独立时钟发请求，可以暴露过载队列增长。
Closed-loop 固定客户端并发，完成后才继续发，服务变慢会自动降低 offered load。
前者适合到达率容量，后者适合固定并发体验。
两者不能相互冒充。

Little's Law 在稳定且边界一致时给出：

$$
L=\lambda W.
$$

有效到达率、系统内请求数和逗留时间必须来自同一边界。
它不提供 p99，也不适用于机械解释持续增长的队列。

## 4. 真实负载比固定 shape 更重要

保留 ISL/OSL 联合分布、prefix、多轮、burst、取消和慢客户端。
只给平均输入长度会隐藏超长请求。
`ignore_eos` 可用于控制实验，但改变真实结束分布。
冷缓存、热缓存和长期 churn 应分别测量。

每轮冻结 artifact、commit、硬件、backend 和并行参数。
保存原始逐请求 trace，让最终表格可以重新计算。

## 5. 从系统到 kernel 的定位顺序

```text
request trace
  → actual dispatch/backend
  → Nsight Systems timeline
  → hotspot reproduction
  → Nsight Compute counters
  → one-hypothesis change
  → end-to-end A/B
```

Nsys 判断 CPU/GPU/通信如何交错。
NCU 解释热点的实际流量、寄存器、spill 和执行瓶颈。
不能把重叠 span 相加后当作 wall time。
Profiler 开销与 replay 也不能当作正常服务时间。

## 6. 收益需要上限与反例

Amdahl 公式为：

$$
Speedup=\frac{1}{(1-f)+f/s}.
$$

占 4% 的 kernel 即使快 2 倍，总收益上限也只有约 2%。
融合可能增加 spill，TP 可能增加通信，缓存可能制造热点。
优化报告需要指出无收益和退化区域。

## 7. 上线门禁

门禁同时包括 correctness、质量、SLO、稳定性和成本。
相对回归阈值不能替代绝对业务 SLO。
成本应按合格请求或 token 归一，并包括 warm pool、网络和失败工作。

PR、nightly 与 release 使用不同大小的验证矩阵。
交错 A/B 降低温度、频率和缓存的顺序偏差。
Shadow/canary 后逐级放量，artifact 与运行时一起可回滚。

## 8. 学习验收

- 从原始时间戳算 TTFT、ITL、TPOT 和 E2E。
- 解释高 TPS 低 goodput 的数例。
- 用 Little's Law 计算平均驻留，说明适用条件。
- 从 timeline 判断 CPU、通信或 kernel 是否在关键路径。
- 用 Amdahl 给优化上限。
- 设计同时保护质量和 p99 的门禁。

每篇均有可运行 CPU 小算例和详细追问答案。

## 9. 来源与边界

对应宝典 `PERF-001—020`、`KERN-023—030`、`PROD-002/012/015/016/020`。
同时覆盖 `CODE-018`、`DESIGN-010/013/015` 和项目证据方法。
训练综述关联第 8.1、8.5、8.7 节。

[Nsight Systems 官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 和 [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 用于工具语义核对。
本章不伪造任何 GPU profiler 输出或生产 benchmark。
继续阅读第 9 章，将测量结果用于配额、扩缩容和故障治理。
