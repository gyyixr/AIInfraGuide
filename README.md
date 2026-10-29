# AIInfraGuide（内容持续更新中）

> 从零开始深入理解 AI Infra 的核心全栈技术

**在线浏览：[https://caomaolufei.github.io/AIInfraGuide/](https://caomaolufei.github.io/AIInfraGuide/)**

AI Infra 正在成为大模型时代最关键的工程能力之一。本项目系统梳理从 GPU 硬件到分布式训练、从 CUDA 编程到推理优化的完整技术栈，帮助工程师构建扎实的 AI 基础设施知识体系。同时提供了面试宝典模块（共收录 180+ 场面试真题，覆盖 60+ 家公司）。

<div align="center">
  <img src="https://img.shields.io/github/stars/caomaolufei/AIInfraGuide.svg?style=flat" alt="Stars" />
  <img src="https://img.shields.io/github/forks/caomaolufei/AIInfraGuide.svg?style=flat" alt="Forks" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome" />
</div>

<img src="https://caomaolufei.github.io/AIInfraGuide/images/AIinfraGuideWeb.png" alt="AIinfraGuideWeb" style="max-width: 100%; display: block; margin: 0 auto;" />

<img src="https://caomaolufei.github.io/AIInfraGuide/images/AIinfraGuideWeb1.png" alt="AIinfraGuideWeb1" style="max-width: 100%; display: block; margin: 0 auto;" />

---

<br>

## 🚀 为什么需要 AIInfraGuide

大模型时代，AI 基础设施（AI Infra）已经成为支撑训练、推理和服务的核心技术底座。然而，这个领域有一个显著的矛盾——技术迭代极快，但系统化的中文学习资料却严重匮乏。

很多工程师在学习 AI Infra 时面临相似的困境：

- CUDA 编程的入门资料散落各处，缺乏从基础到算子优化的完整路径
- 分布式训练涉及 DDP、FSDP、3D 并行等众多概念，不知道该从哪里开始
- 推理优化技术（PagedAttention、量化、Speculative Decoding）发展迅猛，难以跟上节奏
- 性能分析工具（Nsight Systems、Nsight Compute）功能强大，但上手门槛不低

AIInfraGuide 正是为了解决这些问题而创建的——一个**开源、系统、面向实践**的 AI Infra 知识库，帮助工程师构建从硬件到软件、从训练到推理的完整知识体系。

<br>

## 📖 AIInfraGuide 内容体系

知识库围绕 **5 大学习模块 + 2 个辅助板块 + 面试宝典**，覆盖 AI Infra 工程师需要掌握的关键技术栈：

| 板块 | 涵盖内容 |
|---|---|
| **AIInfra 学习路线** | 系统化的学习路径、知识图谱、推荐资源 |
| **模块一：前置知识** | 编程语言基础、数学基础、Transformer 架构、PyTorch 框架、GPU 硬件概论、集合通信基础 |
| **模块二：CUDA 编程与算子优化** | CUDA 编程入门、性能优化基础、Reduce/GEMM/Softmax/Attention 经典算子实现、AI 编译器、性能分析工具链 |
| **模块三：分布式训练** | 分布式训练总论、优化器、数据并行(DP/DDP/FSDP)、ZeRO 系列、张量并行与序列并行、流水线并行、3D 并行、训练框架实战 |
| **模块四：推理优化** | LLM 推理基础、推理引擎核心技术、主流推理框架(vLLM 等)、量化、Speculative Decoding、PD 解耦架构、性能分析与 Benchmark |
| **模块五：LLM RL 后训练** | PPO、GAE、GRPO、DPO、奖励工程、多轮 Agent、在线蒸馏、RL 框架、训练推理协同、数值实验 |
| **性能分析** | Nsight Systems/Compute、Roofline 模型、Profiling 实战 |
| **面试宝典** | 目前共收录 180+ 场面试真题，覆盖 60+ 家公司，按梯队分类组织，助你高效备战拿下心仪 Offer |

每篇文章都遵循「先白话后术语」的写作原则——先用通俗的语言解释"是什么、为什么需要"，再给出严谨的技术细节，确保读者既看得懂也学得对。

<br>

## 🗺️ AIInfra 学习路线

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 1 | [AI Infra 学习路线](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra学习路线/) | AI Infra 技术栈全貌、系统化学习路径与知识图谱 |

<br>

## 📚 模块一：前置知识

涵盖 GPU 架构、编程语言基础、数学基础、Transformer 架构、PyTorch 框架和集合通信等核心前置知识，为后续深入 AI Infra 打好坚实基础。

| 章节 | 主要内容 |
|------|----------|
| 第 1 章 编程语言基础 | Python 进阶、C/C++ 核心、Linux 与开发环境 |
| 第 2 章 数学基础 | 线性代数、概率论与统计、微积分 |
| 第 3 章 Transformer 架构详解 | Self-Attention、前馈网络、位置编码、归一化层、模型架构变种 |
| 第 4 章 PyTorch 框架 | Tensor 与自动微分、Module 与训练流程、调试与性能分析 |
| 第 5 章 GPU 硬件概论 | GPU 架构总览、存储层次、主流 GPU 规格对比、互联拓扑 |
| 第 6 章 集合通信基础 | 通信原语、通信算法、NCCL |

**已更新文章：**

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 第3章 | [🔥 Transformer 架构：快速入门篇](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/transformer架构快速入门/) | Encoder-Decoder 结构、Self-Attention 机制入门 |
| 3.1 | [AI Infra 工程师为什么必须懂 Transformer](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/31-ai-infra工程师为什么必须懂transformer/) | 从 Infra 视角理解 Transformer 的计算与显存特性 |
| 3.2 | [Transformer 全貌及代码实现](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/32-transformer全貌及代码实现/) | 完整 Transformer 架构拆解与 PyTorch 实现 |
| 3.3 | [Self-Attention机制深入理解](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/33-self-attention机制深入理解/) | Self-Attention 是 Transformer 的心脏，也是当代大模型中计算量最集中、优化手段最丰富的模块 |
| 3.4 | [Transformer前馈网络ffn深入理解](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/34-transformer前馈网络ffn深入理解) | 前馈网络 FFN 负责对每个 token 进行独立的非线性变换，是模型记忆知识和深度推理的核心载体 |
| 3.5 | [Transformer位置编码深入理解](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/35-transformer位置编码深入理解) | 位置编码是 Transformer 架构中一个看似不起眼却至关重要的组件 |
| 3.6 | [LayerNorm与残差连接深入理解](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/36-layernorm与残差连接深入理解) | LayerNorm 和 残差连接 决定了大模型能不能学会 |
| 3.7 | [Transformer Decoder Block完整解析](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/37-transformer-decoder-block完整解析) | 大语言模型的核心计算单元是 Transformer Decoder Block |
| 3.8 | [从Transformer到LLM自回归生成深入理解](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解/) | 从宏观视角理解 Transformer 到 LLM 自回归生成的计算流程 |
| 3.9 | [Tokenization与词嵌入](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/39-tokenization与词嵌入/) | 一段文字变成模型输入之前，需要先经过两道关键变换：Tokenization（分词） 和 Embedding（词嵌入） |
| 第4章 | [🔥 PyTorch框架入门](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/pyroch/pytorch框架入门/) | PyTorch 是当前大模型训练和推理的事实标准框架 |
| 第5章 | [🔥 GPU基础知识：从硬件架构到AI计算](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics/) | CPU vs GPU、SM 架构、显存层级、Tensor Core |
| 5.1 | [NVIDIA GPU 架构演进：从 Volta 到 Blackwell](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/nvidia-gpu-evolution/) | V100 → A100 → H100 → B200 架构演进 |
| 第6章 | [🔥 集群通信网络与NCCL：分布式训练的通信骨架](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/collective-communication-primer/) | 从通信原语到 Ring AllReduce 算法，再到 NCCL 实战，快速建立分布式训练集群通信的完整认知体系 |




<br>

## ⚡ 模块二：CUDA 编程与算子优化

从 CUDA 编程入门到经典算子实现（Reduce、GEMM、Softmax、Attention），再到 AI 编译器和性能分析工具链，系统掌握 GPU 编程与算子优化技术。

| 章节 | 主要内容 |
|------|----------|
| 第 1 章 CUDA 编程入门 | 开发环境搭建、编程模型、内存模型、第一个实用 Kernel |
| 第 2 章 CUDA 性能优化基础 | Warp 与执行模型、内存访问优化、Occupancy 与资源分配、同步与原子操作 |
| 第 3 章 经典算子实现 - Reduce | 朴素实现、共享内存+树形归约、Warp Shuffle 优化、多级归约 |
| 第 4 章 经典算子实现 - GEMM | 矩阵乘法基础、Shared Memory Tiling、进一步优化、与 cuBLAS 对比 |
| 第 5 章 经典算子实现 - Softmax 与算子融合 | Softmax 数值稳定实现、Online Softmax、算子融合 |
| 第 6 章 Attention 算子 | FlashAttention V1/V2/V3、Decode 阶段加速、PagedAttention CUDA 实现 |
| 第 7 章 AI 编译器 | Triton、torch.compile、TVM/XLA 概述 |
| 第 8 章 性能分析工具链 | Nsight Systems/Compute、PyTorch Profiler |

**已更新文章：**

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 快速入门| [✅ CUDA编程快速入门指南 🔥](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/cuda编程入门指南/) | CUDA 编程模型、线程层级、内存模型与 Kernel 编写 |
| 第 1 章| CUDA 编程入门 | 开发环境搭建、编程模型、内存模型、第一个实用 Kernel |
| 1.1 | [CUDA 开发环境搭建](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/11-cuda开发环境搭建) | 从零搭建一套完整的 CUDA 开发环境，包括驱动安装、Toolkit 配置、编译工具链和 IDE 集成 |
| 1.2 | [CUDA 编程模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型) | 深入理解 CUDA 的 Grid/Block/Thread 三级线程层次结构、线程索引计算方法和 Kernel 启动配置策略 |
| 1.3 | [CUDA 内存模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/13-cuda内存模型) | 本文详解全局内存、共享内存、寄存器、常量内存和统一内存的特性、适用场景及优化技巧 |
| 1.4 | [第一个实用 Kernel：向量加法](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/14-第一个实用kernel) | 通过一个完整的向量加法案例，走通 CUDA 编程的全流程 |
| 第 2 章| CUDA 性能优化基础 | Warp 与执行模型、内存访问优化、Occupancy 与资源分配、同步与原子操作 |
| 2.1 | [Warp 与执行模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型) | 深入理解 GPU 最核心的执行单元——Warp |
| 2.2 | [内存访问优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/22-内存访问优化) | 内存访问效率是 CUDA Kernel 性能最大的杠杆 |
| 2.3 | [Occupancy 与资源分配](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/23-occupancy与资源分配) | Occupancy 衡量 SM 上实际活跃 Warp 数与理论最大值的比例，是调优 CUDA Kernel 的核心指标之一 |
| 2.4 | [同步与原子操作](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/24-同步与原子操作) | 正确的同步机制是编写无 Bug 并行程序的基础 |
| 第 3 章| 经典算子实现 - Reduce | 朴素实现、共享内存+树形归约、Warp Shuffle 优化、多级归约 |
| 3.1 | [CUDA Reduce算子优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/31-cuda-reduce算子优化) | Reduce（规约）是 GPU 编程中最基础、也最能体现并行思维的算子之一 |
| 第 4 章| 经典算子实现 - GEMM | 矩阵乘法基础、Shared Memory Tiling、进一步优化、与 cuBLAS 对比 |
| 4.1 | [CUDA GEMM算子性能优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化) | 从朴素实现出发，逐步Block-Warp-Thread三级Tiling优化、向量化访存、Bank Conflict 消除、双缓冲等优化手段，带你系统掌握 CUDA GEMM 优化的完整方法论 |
| 第 5 章| 经典算子实现 - Softmax 与算子融合 | Softmax 数值稳定实现、Online Softmax、算子融合 |
| 5.1 | [CUDA Softmax 朴素实现优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/51-cuda-softmax朴素实现优化) | 从朴素实现出发，逐步引入 Safe Softmax、Block 级并行、Warp Shuffle、向量化访存等优化手段 |
| 5.2 | [CUDA Online Softmax 实现优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现) | 逐步实现并优化 Online Softmax 的 CUDA Kernel，这也是理解 FlashAttention 的核心前置知识 |
| 第 6 章| Attention 算子 | FlashAttention V1/V2/V3、Decode 阶段加速、PagedAttention CUDA 实现 |
| 6.1 | [FlashAttention V1详解](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解) | 本文深入剖析 FlashAttention V1 的核心原理与实现细节，理解如何通过 Tiling + Online Softmax 将 Attention 的额外显存占用从 $O(N^2)$ 降至 $O(N)$，同时大幅减少 HBM 访问量，实现无精度损失的加速 |
| 6.2 | [FlashAttention V2详解](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解) | 本文深入剖析 FlashAttention V2 相比 V1 的核心改进：调换内外循环顺序、优化线程块内的工作分配、减少非矩阵乘运算 |
| 第 7 章| AI 编译器 | Triton、torch.compile、TVM/XLA 概述 |
| 7.1| [Triton 编程模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/71-triton编程框架)  | 分析 Triton 的编程模型，并结合 Softmax 算子实现介绍了 Triton 基础语法 |
| 第 8 章| 性能分析工具链 | Nsight Systems/Compute、PyTorch Profiler |


<br>

## 🌐 模块三：分布式训练

从分布式训练总论与集合通信原语出发，深入优化器、数据并行、ZeRO 系列、张量并行、流水线并行、混合精度与显存优化、长序列与上下文并行、MoE 并行，最终通过 3D 并行与混合并行策略串联全部知识。

| 章节 | 主要内容 |
|------|----------|
| 第 1 章 分布式训练总论 | 为什么需要分布式训练、训练显存账本、五大并行策略全景、环境搭建与 DDP 启动 |
| 第 2 章 集合通信原语 | AllReduce/ReduceScatter/AllGather/All-to-All 语义、Ring 算法、通信量量化分析 |
| 第 3 章 优化器 | SGD/Adam/AdamW 演进、优化器状态显存开销分析、大 Batch 优化器 LAMB/LARS |
| 第 4 章 数据并行 | DataParallel、DistributedDataParallel、FSDP |
| 第 5 章 ZeRO 系列 | ZeRO-1/2/3 切分策略与通信代价、ZeRO-Offload/Infinity 异构内存卸载 |
| 第 6 章 张量并行与序列并行 | Megatron Column/Row Parallel Linear、通信插入位置、序列并行激活显存优化 |
| 第 7 章 流水线并行 | Bubble 问题本质、GPipe、1F1B、Interleaved 调度 |
| 第 8 章 混合精度与显存优化 | FP16/BF16/FP8 混合精度、梯度累积、Activation Checkpointing |
| 第 9 章 长序列训练与上下文并行 | Attention 的 O(s²) 困境、Ring Attention、DeepSpeed-Ulysses、Megatron Context Parallel |
| 第 10 章 MoE 并行 | Router 机制、Expert Parallelism 的 All-to-All 通信、EP×DP×TP×PP 组合与负载均衡 |
| 第 11 章 3D 并行与混合并行策略 | TP×PP×DP×EP×CP 拓扑设计、通信域映射、rank 编排、集群选型 |

**已更新文章：**

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 第 1 章| 分布式训练总论 | 为什么需要分布式训练、训练显存账本、五大并行策略全景、环境搭建与 DDP 启动 |
| 1.1 | [分布式训练总论：显存账本与五大并行策略全景](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/11-分布式训练总论/) | 手算训练显存账本，总览五大并行策略全景与选择原则 |
| 1.2 | [环境搭建与分布式启动](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/12-环境搭建与分布式启动/) | rank/local_rank/world_size、torchrun 启动与会合机制、NCCL 调试、最小 DDP 脚本 |
| 第 2 章| 集合通信原语 | AllReduce/ReduceScatter/AllGather/All-to-All 语义、Ring 算法、通信量量化分析 |
| 2.1 | [集合通信原语详解](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/21-集合通信原语详解/) | AllReduce/ReduceScatter/AllGather/All-to-All 语义、Ring AllReduce 原理、通信耗时量化 |
| 第 3 章| 优化器 | SGD/Adam/AdamW 演进、优化器状态显存开销分析、大 Batch 优化器 LAMB/LARS |
| 3.1 | [优化器：从 SGD 到 AdamW 与显存开销分析](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/31-优化器原理与显存开销分析/) | 优化器演进、优化器状态显存开销、FP32 主权重的必要性、LARS/LAMB |
| 第 4 章| 数据并行 | DataParallel、DistributedDataParallel、FSDP |
| 4.1 | [数据并行详解](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/41-数据并行详解/) | DP/DDP/FSDP 演进、梯度同步机制、通信量与显存账本的数学推导 |
| 4.2 | [PyTorch 数据并行从原理到实战](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/42-pytorch-数据并行从原理到实战/) | 通信原语、DDP、FSDP 参数分片、多机多卡实战，附完整代码示例 |

<br>

### 本次深入补充（2026-10-30）

分布式训练现包含 **11 章导读、23 篇详解**，新增以下 17 篇专题。后七章补齐逐步推导、可手算例子、源码入口、实践边界和自测答案，同时修订旧文中的通信字节、精度状态与拓扑假设。

| 专题 | 内容 |
|---|---|
| [2.2 Ring 逐轮推导与通信性能分析](docs/guides/模块三-分布式训练/2.2-Ring逐轮推导与通信性能分析.md) | 手算四 rank Ring 的六轮发送、接收与累加，建立字节和时间模型，再分析 bucket、异步完成语义、带宽测试和通信排障 |
| [3.2 梯度累积与有效 Batch 更新](docs/guides/模块三-分布式训练/3.2-梯度累积与有效Batch更新.md) | 推导不等样本数下的全局 loss 与 DDP 缩放，逐步计算梯度累积、裁剪和 AdamW 更新，区分 micro-step、optimizer step 与并行 rank |
| [4.3 两进程 DDP 实验与正确性验收](docs/guides/模块三-分布式训练/4.3-两进程DDP实验与正确性验收.md) | 无需 GPU 地运行真实 Gloo 集合通信、DDP 梯度累积与 AdamW 更新，对照单进程 oracle 检查不等样本数、全局裁剪、状态恢复和 rank 一致性 |
| [5.1 ZeRO 分片与显存通信账本：每份状态由谁负责](docs/guides/模块三-分布式训练/5.1-ZeRO分片与显存通信账本.md) | 从四参数更新推导 ZeRO-1/2/3，统一参数、梯度与优化器字节口径，手算 7B 模型的静态下限、瞬时峰值及每 rank 通信量 |
| [5.2 FSDP2 与 Offload 工程实践：让状态生命周期可观察](docs/guides/模块三-分布式训练/5.2-FSDP2与Offload工程实践.md) | 将 ZeRO 账本映射到 fully_shard、DTensor、参数组与预取，提供最小训练骨架，并量化 CPU/NVMe 卸载的内存、带宽和恢复成本 |
| [6.1 张量并行的前向、反向与通信：从两层矩阵乘法推导](docs/guides/模块三-分布式训练/6.1-张量并行的前向反向与通信.md) | 手算 Column/Row Parallel Linear 的完整前反向，区分分片与部分和，推导 Transformer TP 的通信位置、字节量及扩展边界 |
| [6.2 序列并行与并行线性层实践：在算子之间保持正确布局](docs/guides/模块三-分布式训练/6.2-序列并行与并行线性层实践.md) | 从 TP 的残余激活复制推导 Megatron 风格 SP，跟踪 AllGather/ReduceScatter 前反向、RMSNorm 梯度、GQA 约束与 PyTorch DTensor 方案 |
| [7.1 流水线调度与 Bubble 推导：把每个时隙画清楚](docs/guides/模块三-分布式训练/7.1-流水线调度与Bubble推导.md) | 从层间依赖推导 GPipe 与同步 1F1B，用逐时隙时间线核对气泡、激活存活量和梯度累积，区分利用率公式与真实吞吐 |
| [7.2 交错流水线与负载均衡：把省下的 Bubble 变成吞吐](docs/guides/模块三-分布式训练/7.2-交错流水线与负载均衡.md) | 推导虚拟 stage 的理想收益和消息代价，手算分层负载、P2P 数据量与瓶颈，结合 Megatron 调度源码建立可复核的优化步骤 |
| [8.1 混合精度与优化器数值链路](docs/guides/模块三-分布式训练/8.1-混合精度与优化器数值链路.md) | 从浮点间距出发，逐步跟踪前向、反向、梯度累积、Adam 状态和主权重更新，区分 autocast 与显式 BF16 参数并复算显存账本 |
| [8.2 激活重计算与显存生命周期](docs/guides/模块三-分布式训练/8.2-激活重计算与显存生命周期.md) | 从反向所需张量和峰值重叠出发，推导 checkpoint 分段模型、重算时间代价、梯度累积及选择性保存，提供带 Dropout 的一致性实验 |
| [9.1 长序列训练与上下文并行](docs/guides/模块三-分布式训练/9.1-长序列训练与上下文并行.md) | 手算 32K Attention 的显存与计算量，追踪 Ring、AllGather 与 Ulysses 的张量布局和通信字节，再分析 CP 与 DP/TP、mask、标签和全局 loss 的接口 |
| [9.2 RingAttention 与长序列工程实践](docs/guides/模块三-分布式训练/9.2-RingAttention与长序列工程实践.md) | 逐轮追踪四 rank 的 KV 环传，手推 online softmax 与反向梯度，再分析因果负载、通信重叠、全局随机数和真实多卡验证路径 |
| [10.1 MoE 路由与专家并行：逐个 token 走完 Dispatch 与 Combine](docs/guides/模块三-分布式训练/10.1-MoE路由与专家并行.md) | 用六 token、四专家、两设备的完整例子推导 Top-2 路由、All-to-All split、专家矩阵形状、输出归并和反向传播，区分 EP、ETP 与专家数据副本 |
| [10.2 MoE 负载均衡与通信优化：从路由直方图到关键路径](docs/guides/模块三-分布式训练/10.2-MoE负载均衡与通信优化.md) | 手算专家负载、容量与辅助损失，拆解 Grouped GEMM、permutation、All-to-All 和通信重叠，建立不会混淆质量与吞吐的 MoE 优化流程 |
| [11.1 多维并行拓扑与 Rank 编排：把每张卡放进正确的通信组](docs/guides/模块三-分布式训练/11.1-多维并行拓扑与Rank编排.md) | 手列 8 卡 TP×PP×DP 通信组，解释 CP 和专家并行的不同网格，再比较 64 卡拓扑、global batch 与实际通信域 |
| [11.2 训练性能建模与可靠恢复：把并行配置变成可验收的账本](docs/guides/模块三-分布式训练/11.2-训练性能建模与可靠恢复.md) | 逐项计算混合精度状态、分片内存、通信下界和流水线空泡，统一 token 吞吐与 MFU 口径，并设计包含 optimizer、RNG 和数据进度的恢复实验 |

本地浏览：[分布式训练模块](http://localhost:4321/AIInfraGuide/distributed/)。配套 [实验说明](labs/distributed/README.md) 与 [验证记录](labs/distributed/VALIDATION.md)：

```bash
npm run test:distributed
npm run lab:distributed
# 在具备 PyTorch/Gloo 的 Python 环境中：
npm run lab:ddp
```

纯标准库脚本验证数学机制；DDP 实验真实运行两个 CPU/Gloo 进程，不将其写作多 GPU 或生产性能验证。指定解释器的方法见实验说明。

<br>

## 🚀 模块四：推理优化

以 **vLLM** 为贯穿主线，覆盖 LLM 推理基础、推理引擎核心技术、nano-vllm 源码解读、量化、Speculative Decoding、分布式推理、PD 解耦架构、生产级服务特性、性能分析、生产部署运维，以及端到端选型实战。

| 章节 | 主要内容 |
|------|----------|
| 第 1 章 LLM 推理基础 | Prefill/Decode 两阶段、KV Cache、性能指标、Roofline 瓶颈分析、推理链路 |
| 第 2 章 推理引擎核心技术 | PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill、Attention 后端与图优化 |
| 第 3 章 nano-vllm 源码解读 | 以约 1200 行极简引擎 nano-vllm 为主线，逐模块读懂连续批处理、PagedAttention、FlashAttention、张量并行与 CUDA Graph |
| 第 4 章 量化 | 量化基础、SmoothQuant、GPTQ/AWQ、FP8、NVFP4/MXFP4、KV Cache 量化 |
| 第 5 章 Speculative Decoding | Draft+Verify 原理、N-gram/Suffix、Medusa/EAGLE/MTP、vLLM 投机解码实战 |
| 第 6 章 分布式推理与大模型部署 | 张量并行、流水线并行、数据/专家并行、Ray 多节点、vLLM 分布式部署 |
| 第 7 章 PD 解耦架构 | 混合 Batching 干扰、DistServe/Splitwise、KV Cache 传输、Goodput 与 SLO 调度 |
| 第 8 章 性能分析与 Benchmark | 推理指标体系、vllm bench/GenAI-Perf、Nsight、性能回归门禁 |
| 第 9 章 生产部署与运维 | 容器化与 K8s 部署、可观测性、自动扩缩容、负载均衡、容量规划 |
| 第 10 章 端侧推理 | 端侧软硬件栈、异构执行、运行时选型、量化、内存优化与 Benchmark |

**已更新文章：**

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 第 1 章 | LLM 推理基础 | Prefill/Decode 两阶段、KV Cache、性能指标、Roofline 瓶颈分析、推理链路 |
| 1.1 | [LLM 推理基础](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础/) | Prefill/Decode 两阶段、KV Cache 显存账本、TTFT/TPOT 指标，用 Roofline 解释 Decode 为何 Memory Bound |
| 1.2 | [vLLM 快速入门](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/12-vllm快速入门/) | 从安装到部署你的第一个 LLM 推理服务，离线批量推理与在线 OpenAI 兼容服务 |
| 第 2 章 | 推理引擎核心技术 | PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill、Attention 后端与图优化 |
| 2.1 | [PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) | 借鉴虚拟内存分页思想，用 Block Table 消除 KV Cache 碎片化，支持前缀共享与 Copy-on-Write |
| 2.2 | [Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) | Iteration-level Scheduling 让请求随到随拼、完成即退，把 GPU 利用率从三成拉到八成以上 |
| 2.3 | [Prefix Cache 与 RadixAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) | vLLM 基于 Hash 的自动前缀缓存，SGLang 用 Radix Tree 做更高效的前缀共享 |
| 2.4 | [Chunked Prefill 与统一调度](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) | 切块 Prefill 消除对 Decode 的干扰，vLLM V1 用统一 Token 预算调度器抹平 Prefill/Decode 边界 |
| 2.5 | [Attention 后端与图优化](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) | 可插拔 Attention 后端与 CUDA Graph、torch.compile 消除 Decode 阶段的 CPU 启动开销 |
| 第 3 章 | nano-vllm 源码解读 | 以约 1200 行极简引擎 nano-vllm 为主线，逐模块读懂连续批处理、PagedAttention、FlashAttention、张量并行与 CUDA Graph |
| 3.1 | [项目全景与四层架构](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/31-项目全景与四层架构/) | nano-vllm 定位、目录结构，与接口/引擎/调度/执行四层架构及端到端数据流 |
| 3.2 | [配置、入口与一次 generate 的全流程](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/32-配置入口与推理主循环/) | Config/SamplingParams、LLM 对 LLMEngine 的封装，以及 generate→add_request→step→postprocess 主循环 |
| 3.3 | [模型骨架与核心算子](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/33-模型与核心算子/) | RMSNorm/RoPE/SwiGLU、并行 Linear 族与张量并行、词表并行 Embedding/LMHead、Attention 两条路径 |
| 3.4 | [KV Cache 与 BlockManager](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/34-kvcache与blockmanager/) | Block 结构、PagedAttention 分块、基于 xxhash 的前缀缓存、块的分配回收与 Sequence 状态机 |
| 3.5 | [Scheduler 与连续批处理](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/35-scheduler与连续批处理/) | Waiting/Running 双队列、Prefill/Decode 分离调度、Token Budget、chunked prefill 与抢占重算 |
| 3.6 | [ModelRunner、CUDA Graph 与性能基准](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/36-modelrunner引擎循环与性能/) | 序列张量化、全局 Context、KV Cache 显存切分、CUDA Graph 捕获回放，以及官方基准与 vLLM 对比 |
| 第 4 章 | 量化 | 量化基础、SmoothQuant、GPTQ/AWQ、FP8、NVFP4/MXFP4、KV Cache 量化 |
| 4.1 | [量化基础](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | Scale 与零点、对称与非对称、位宽与格距、量化粒度、校准与 PTQ/QAT |
| 4.2 | [W8A8 与 SmoothQuant](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | 激活离群值、等价缩放与迁移强度 α、INT8 GEMM、LLM Compressor 配方 |
| 4.3 | [Weight-only INT4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | GPTQ 误差补偿、AWQ 激活感知缩放、Marlin 融合 Kernel 与 vLLM 的 Kernel 选择 |
| 4.4 | [KV Cache 量化](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | KIVI 分组与残留缓存、FP8 KV 的 Scale 校准与跳层、长上下文验证 |
| 4.5 | [FP8 与 NVFP4/MXFP4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | E4M3/E5M2、FP8 Scale、E2M1 微块缩放、NVFP4 与 MXFP4 对比及硬件执行条件 |
| 4.6 | [量化选型与 vLLM 实战](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | 基于 v0.32.0 的按瓶颈选型、在线量化入口、FP16/AWQ 吞吐与生成质量对照 |
| 第 5 章 | Speculative Decoding | Draft+Verify 原理、N-gram/Suffix、Medusa/EAGLE/MTP、vLLM 投机解码实战 |
| 5.1 | [投机解码核心原理](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | Draft + Verify、接受与残差采样的保分布证明、one-hot Draft、KV 回退 |
| 5.2 | [Draft 模型与无模型方案](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | 独立 Draft 与 PARD、Tokenizer 兼容、N-gram/Suffix、接受率与候选覆盖 |
| 5.3 | [Medusa、EAGLE 与 MTP](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | 多头预测、Tree Attention、EAGLE-2/3、DeepSeek-V3 MTP、Block Verification |
| 5.4 | [收益边界与限制](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | 提交长度与轮耗时账本、投机长度递减、Dynamic SD、并发、量化与流式延迟 |
| 5.5 | [vLLM 投机解码实战](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | 基于 v0.32.0 的基线/N-gram/Draft/EAGLE-3 对照、逐请求接受指标与 Benchmark |
| 第 6 章 | 分布式推理与大模型部署 | 张量并行、流水线并行、数据/专家并行、Ray 多节点、vLLM 分布式部署 |
| 第 7 章 | PD 解耦架构 | 混合 Batching 干扰、DistServe/Splitwise、KV Cache 传输、Goodput 与 SLO 调度 |
| 第 8 章 | 性能分析与 Benchmark | 推理指标体系、vllm bench/GenAI-Perf、Nsight、性能回归门禁 |
| 第 9 章 | 生产部署与运维 | 容器化与 K8s 部署、可观测性、自动扩缩容、负载均衡、容量规划 |
| 第 10 章 | 端侧推理 | 端侧软硬件栈、异构执行、运行时选型、量化、内存优化与 Benchmark |
| 10.1 | [端侧推理基础：从模型导出到异构硬件执行](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-端侧推理/121-端侧推理基础/) | 端侧约束、软硬件栈、运行时选型、量化与内存优化、Benchmark，以及 ExecuTorch/XNNPACK 最小实例 |

<br>

## 🧠 模块五：LLM RL 后训练

沿用分布式训练模块的「章节导读 + 专题详解」结构，包含 **10 章导读、20 篇详解**。每篇结合直觉、公式、数值或系统算例、工程排查、自测答案和一手资料。

本地站点入口：[LLM RL 后训练](http://localhost:4321/AIInfraGuide/rl/)。Markdown 源文档位于 [模块五目录](docs/guides/模块五-LLM-RL后训练/)。

| 章节 | 专题 |
|---|---|
| 第 1 章 后训练全景 | SFT/RLHF/RLVR、轨迹张量与策略梯度 |
| 第 2 章 PPO 与价值学习 | 裁剪目标、GAE、一轮 PPO 数据流 |
| 第 3 章 GRPO 与 RLVR | 组内优势、RLOO、DAPO/Dr. GRPO、长度与稳定性 |
| 第 4 章 偏好优化与 DPO | 偏好数据、奖励模型、DPO 推导与边界 |
| 第 5 章 奖励工程与评估 | 可验证奖励、数据流水线、pass@k、训练诊断 |
| 第 6 章 多轮 Agent 与在线蒸馏 | 工具环境、轨迹 mask、信用分配、OPD |
| 第 7 章 RL 系统架构与框架 | 角色编排、进程通信、verl / ROLL / slime |
| 第 8 章 训练推理协同与性能优化 | 混合部署、权重同步、异步 rollout、吞吐与显存 |
| 第 9 章 可靠训练与故障排查 | Checkpoint、恢复、可复现性、故障定位 |
| 第 10 章 数值实验与学习验收 | 手算目标、CPU RL 闭环、梯度校验、精确恢复 |

配套 [CPU 数值实验](labs/llm_rl/README.md) 无需下载模型或安装 GPU 依赖：

```bash
npm run test:rl
npm run lab:rl
```

实验使用四上下文、四动作策略表，验证数学与状态流转；完整 GPU 框架训练和多机性能未作为已验证结果。

## 💻 本地浏览

需要 Node.js（建议使用当前受支持的 LTS）和 npm；数值实验另需 Python 3.10+。从仓库根目录运行：

```bash
npm ci
npm run site:local
```

打开 **[http://localhost:4321/AIInfraGuide/](http://localhost:4321/AIInfraGuide/)**，新模块为 **[http://localhost:4321/AIInfraGuide/rl/](http://localhost:4321/AIInfraGuide/rl/)**。`site:local` 会依次检查 Astro 类型、构建页面、生成 Pagefind 搜索索引并启动预览；启动期间请等待构建完成。按 `Ctrl+C` 停止。

在 WSL 中启动后，可从 Windows 浏览器访问同一个 localhost 地址。端口 4321 已占用时，先看终端输出的实际端口，或用 `npm run preview -- --host 0.0.0.0 --port 4322` 预览已构建内容。服务绑定 `0.0.0.0`，便于 WSL/主机访问。

日常修改 Markdown 时用 `npm run dev:local` 热更新。搜索索引由构建生成，修改后要执行 `npm run build` 更新搜索；预览模式下修改内容也需要重新构建。URL 必须保留已有 `/AIInfraGuide/` 前缀。

构建后可用 `npm run check:rl-site` 检查新模块的页面覆盖、链接、锚点与公式；详见[交付验证记录](labs/llm_rl/VALIDATION.md)。

<br>

## 🎯 面试宝典

共收录 **180+ 场面试真题**，覆盖 **60+ 家公司**，按梯队分类组织。[在线浏览 →](https://caomaolufei.github.io/AIInfraGuide/interview)

涵盖公司包括字节跳动、阿里巴巴、腾讯、百度、快手、美团、蚂蚁、英伟达、MiniMax、蔚来、小鹏、理想等。

<img src="https://caomaolufei.github.io/AIInfraGuide/images/AIinfraGuideWeb2.png" alt="AIinfraGuideWeb2" style="max-width: 100%; display: block; margin: 0 auto;" />

<br>

**投递总览：**

| 梯队 | 公司 | 主要 AI Infra 岗位 |
|------|------|------|
| **T0 大厂** | 字节跳动、阿里巴巴、腾讯、百度 | AI Infra 工程师、高性能计算研发、推理优化工程师、**分布式训练框架** |
| **T1 大厂/独角兽** | 快手、拼多多、美团、蚂蚁、OPPO、华为、蔚来 | AI Infra、高性能计算、大模型推理优化、AI 平台开发 |
| **T2 AI 独角兽** | MiniMax、阶跃星辰、智谱AI、面壁智能、月之暗面 | AI Infra、大模型算法 (偏 Infra)、推理系统开发、Agent Infra |
| **T3 芯片/硬件厂商** | 英伟达、摩尔线程、海光、寒武纪、壁仞科技、飞腾 | 算子开发、CUDA 优化、GPU 软件工程师、AI 编译器、高性能计算 |
| **T4 自动驾驶/车企** | 小鹏汽车、蔚来、理想、卓驭(大疆车载)、小马智行、元戎启行 | AI Infra、高性能计算、大模型推理优化、AI 平台 |
| **T5 其他** | 科大讯飞、网易、海康威视、联想、猿辅导、好未来等 | AI Infra、高性能计算、推理引擎开发、大模型算法 |

> 注：分布式训练相关的岗位特别少，一般只有做大模型的大厂才有，所以大家在求职准备过程中，可以重点关注算子优化、推理优化、推理框架这类岗位，训练相关的技术点了解即可，不必深钻。

<br>

### 各梯队面试侧重点

| 梯队 | 考察侧重 |
|------|----------|
| **T0 大厂**（字节/阿里/腾讯/百度） | 结构化面试 2-3 轮，项目深挖 + 模型架构 + 推理/训练优化 + C++ 八股 + LeetCode Medium |
| **T1 大厂/独角兽**（快手/美团/蚂蚁等） | 与 T0 类似，推理优化问得多，部分公司重视系统设计 |
| **T2 AI 独角兽**（MiniMax/阶跃/智谱） | 偏研究导向，深挖 MoE 路由优化、RLHF 细节、前沿 paper，传统八股较少 |
| **T3 芯片/硬件**（英伟达/壁仞/寒武纪） | 重 GPU 架构、CUDA 编程、HPC 基础，手写 kernel 频率最高 |
| **T4 车企/自动驾驶**（蔚来/大疆车载/小鹏） | 重 C++ 功底、推理部署（TensorRT/TVM/量化）、边缘实时性能，LLM 问得少 |
| **T5 其他**（海康/科大讯飞等） | 难度适中，C++ 和推理部署为主 |


<br>

**投递时间线：**

| 时间段 | 投递策略 |
|--------|----------|
| **3-4月** | 暑期实习提前批（字节、阿里、腾讯、快手）|
| **5-6月** | 暑期实习正式批 + 日常实习（美团、拼多多、中小厂）|
| **7-8月** | 秋招提前批（字节 AML、百度、华为）|
| **9-10月** | 秋招正式批（全面投递）|
| **11-12月** | 秋招补录 + 春招准备 |
| **次年2-4月** | 春招（HC 较少，竞争激烈）|

---

<br>

## 🤝 欢迎参与贡献

AIInfraGuide 是一个开源项目，欢迎通过以下方式参与共建：

- **提交 Issue**：发现错误、提出建议，或者告诉我们你希望看到的主题
- **贡献 PR**：分享你的实践经验、补充技术细节、改进现有内容
- **Star & Share**：如果觉得有帮助，请在 [GitHub](https://github.com/caomaolufei/AIInfraGuide) 上给个 Star，让更多人发现这个项目

💡 **提示**：如果你不确定从哪里开始，推荐先阅读知识库中的「AIInfra 学习路线」，它会帮你梳理一条清晰的学习路径。

让我们一起构建 AI Infra 社区的知识基础设施。

<div id="contribute"></div>

<div align='center'>
<a href="https://star-history.dera.page/#caomaolufei/AIInfraGuide&type=Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://star-history.dera.page/svg?repos=caomaolufei/AIInfraGuide&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://star-history.dera.page/svg?repos=caomaolufei/AIInfraGuide&type=Date" />
    <img width=400 height=300 alt="Star History Chart" src="https://star-history.dera.page/svg?repos=caomaolufei/AIInfraGuide&type=Date" />
  </picture>
</a>
</div>

## 联系方式

欢迎通过以下方式与我交流：

- 公众号: [《AI Infra Guide》](https://mp.weixin.qq.com/s/oxUBDTd-b1aZnh0jGR5tqw)
- 知乎: [草帽路飞](https://zhihu.com/people/zi-you-zi-zai-de-zhan-sheng)
- 社群: （vx： caomaolufei2026）
<img src="https://caomaolufei.github.io/AIInfraGuide/images/wx_qun.png" alt="wx_qun" style="max-width: 50%; display: block; margin: 0 auto;" />

## License

MIT
