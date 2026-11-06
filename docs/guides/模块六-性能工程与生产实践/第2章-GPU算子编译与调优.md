---
title: "第2章：GPU算子编译与调优"
description: "连接warp地址、微架构资源、实现路线、编译guard与CUDA Graph生命周期"
pubDate: 2026-11-07
category: "performance-analysis"
chapter: 2
order: 20
tags: ["性能工程", "训练推理", "实验", "正确性"]
---

## 本章定位

连接warp地址、微架构资源、实现路线、编译guard与CUDA Graph生命周期。
本章将公式、状态机、CPU算例和官方资料结合起来。
它不把理论上界、源码静态证据和硬件实测混成一个结论。

## 学习路径

- [从 GPU 时间线到微架构瓶颈](./2.1-从GPU时间线到微架构瓶颈.md)。
- [编译、CUDA 图与动态 Shape 工程](./2.2-编译CUDA图与动态Shape工程.md)。

先读机制与手算，再运行正文标准库代码。
若继续做GPU或集群实验，按正文矩阵冻结环境并保留原始数据。

## 1. 从地址而不是语法判断访存

同一warp各lane的地址决定覆盖多少sector和事务。
连续访问、stride、misalignment与向量类型必须分开分析。
Shared memory还需考虑bank映射和broadcast特例。
Padding可以减少冲突，也可能增加容量并降低驻留。

## 2. 资源约束与算子形状

Registers、shared、threads和block上限共同限制occupancy。
更高occupancy并不自动更快，spill和ILP可能反向变化。
训练大GEMM、decode skinny GEMM和MoE grouped GEMM形状不同。
Tile、epilogue与pipeline要对真实分布验证。

```text
system hotspot
  → actual kernel
  → address / instruction model
  → counters
  → one change
  → correctness + end-to-end A/B
```

## 3. 实现路线

先检查vendor/framework是否已有满足语义的高效实现。
Triton适合快速tile/fusion，CUDA提供更细控制。
CUTLASS/CuTe提供布局、MMA与pipeline抽象。
TileLang提供tile级算法与调度表达，仍需核对target与生成代码。

语言名字不决定性能。
同一算法在不同架构的原语、编译器和维护成本都不同。
SM86实验与Hopper/Blackwell专用路径必须分开。

## 4. 编译与图的三层边界

计算IR、编译后的kernel与CUDA Graph是不同对象。
Graph break、guard failure和recompile也应分别记录。
动态shape可以减少specialization，但不会消除所有条件。
Bucket减少变体，却引入padding和graph pool占用。

普通capture/replay依赖稳定资源身份。
把Python变量绑定到新tensor不等于更新捕获地址。
取消、卸载和并发replay都需要明确最后访问事件。

## 5. 动手任务

- 用CPU画warp的sector覆盖。
- 比较32与33行stride的bank映射。
- 计算register变化对驻留block的近似限制。
- 核对融合前后的逻辑读写字节。
- 计算shape bucket的padding放大。
- 计算编译回本次数并模拟captured buffer身份。

## 6. 学习验收

能从源码表达式走到实际事务和机器指令证据。
能解释为何fusion、occupancy或graph可能无收益。
能区分编译成功、配置合法、可捕获、数值合格与最快。
能为动态shape建立fallback和生命周期矩阵。

## 7. 建议的证据台账

| 类型 | 保存内容 |
|---|---|
| 版本 | framework commit、依赖与artifact |
| 工作量 | shape/trace/seed、有效与执行口径 |
| 资源 | peak memory、bytes、inflight状态 |
| 时间 | wall/device边界与关键路径 |
| 质量 | reference、任务或训练护栏 |
| 统计 | 重复、配对、分群与原始样本 |
| 边界 | 失败、fallback与未验证硬件 |

每篇都有至少一个完整手算或可运行CPU小例子。
这些代码只验证所写模型，不自动验证真实GPU、网络或存储。
进一步实验应保持同样的时间与数据口径。

## 8. 参考资料与版本规则

本模块整合推理面试宝典的PERF/KERN/DIST/PROD/PROJECT部分。
训练综述补充跨训练的算子、通信、数据与可靠性视角。
Megatron报告补充schedule、optimizer ownership、overlap与公式口径。

Megatron参考源码冻结为48a887fecd01674346f724dcf44f417f455989f0。
Stable0.19是另一层证据，不因包字符串相近就视为同一实现。
SGLang/vLLM固定源码证据沿推理模块第11章。
动态官方文档用于方法与API边界，安装环境仍需独立核验。

## 9. 阅读后的下一步

先选择一个完整语义已明确的算子。
保存其shape、stride、dtype与输入输出别名规则。
用lane地址和tile图预测事务与复用，再用生成代码和counter核对。
若加入compile或graph，另记编译时间、变体数和长期buffer。
这样能区分算子算法改善、codegen改善与提交开销减少。

选择一个真实瓶颈，写出可证伪假设与最小实验。
先确认有效工作和正确性，再衡量关键路径变化。
出现与预测不同的结果时，保留失败并修正模型。
完成时交付可复跑数据、条件与回退，而不是孤立的最佳数字。
