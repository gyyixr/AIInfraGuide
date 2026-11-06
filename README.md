# AIInfraGuide（内容持续更新中）

> 从零开始深入理解 AI Infra 的核心全栈技术

**上游在线浏览：[https://caomaolufei.github.io/AIInfraGuide/](https://caomaolufei.github.io/AIInfraGuide/)** ｜ [更新日志](docs/posts/update-log.md)

本项目系统梳理 AI Infra 的技术栈，从 GPU 硬件、CUDA 编程与算子优化，到分布式训练、推理优化、LLM RL 后训练、性能工程与 DeepSeek 开放技术，帮助工程师建立可以结合源码与实验复核的知识体系。另设面试宝典板块，收录 180+ 场面试真题，覆盖 60+ 家公司。

**当前扩展内容：** 以本仓库和本地站点为准；上游在线站点不代表本 fork 的新增内容已经发布。[进阶导航](docs/guides/训练推理与DeepSeek进阶学习导航.md) · [资料来源与覆盖边界](docs/guides/资料来源版本与覆盖边界.md)

**快速导航：** [学习路线](#-aiinfra-学习路线) · [前置知识](#-模块一前置知识) · [CUDA 编程与算子优化](#-模块二cuda-编程与算子优化) · [分布式训练](#-模块三分布式训练) · [推理优化](#-模块四推理优化) · [LLM RL 后训练](#-模块五llm-rl-后训练) · [性能工程](#-模块六性能工程与生产实践) · [DeepSeek开放技术](#-模块七deepseek-开放技术详解) · [面试宝典](#-面试宝典) · [本地浏览](#-本地浏览) · [参与贡献](#-欢迎参与贡献)

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

## 💡 为什么需要 AIInfraGuide

AI 基础设施（AI Infra）支撑着大模型的训练、推理和服务。这个领域技术迭代很快，系统化的中文学习资料却不多。

很多工程师在学习 AI Infra 时面临相似的困境：

- CUDA 编程的入门资料散落各处，缺乏从基础到算子优化的完整路径
- 分布式训练涉及 DDP、FSDP、3D 并行等众多概念，不知道该从哪里开始
- 推理优化技术（PagedAttention、量化、Speculative Decoding）发展迅猛，难以跟上节奏
- 性能分析工具（Nsight Systems、Nsight Compute）功能强大，但上手门槛不低

AIInfraGuide 是一个**开源、系统、面向实践**的 AI Infra 知识库，按模块把这些内容串成一条从硬件到软件、从训练到推理的学习路径。

<br>

## 📖 AIInfraGuide 内容体系

知识库由 **学习路线 + 7 大学习模块 + 面试宝典** 组成。新增性能工程与 DeepSeek 专题，并把训练、推理、RL 的源码与状态生命周期继续展开：

| 板块 | 涵盖内容 |
|---|---|
| **AIInfra 学习路线** | 系统化的学习路径、知识图谱、推荐资源 |
| **模块一：前置知识** | 编程语言基础、数学基础、Transformer 架构、PyTorch 框架、GPU 硬件概论、集合通信基础 |
| **模块二：CUDA 编程与算子优化** | CUDA 编程入门、性能优化基础、Reduce/GEMM/Softmax/Attention 经典算子实现、AI 编译器、性能分析工具链 |
| **模块三：分布式训练** | 并行基础、HSDP/DistOpt、FP8/FP4、长上下文、数据治理与索引、Megatron源码、可靠性、成本、项目及框架/异构生态 |
| **模块四：推理优化** | LLM 推理基础、推理引擎核心技术、nano-vllm 源码解读、量化、Speculative Decoding、分布式推理、PD 解耦架构、性能分析与 Benchmark、生产部署、端侧推理 |
| **模块五：LLM RL 后训练** | PPO/GAE、GRPO/DPO、奖励、多轮Agent、在线蒸馏、框架、权重同步、长轨迹mask、GSPO与环境队列、数值实验 |
| **模块六：性能工程与生产实践** | Roofline、统计测量、Nsys/NCU、编译与Graph、NCCL/RDMA、NUMA/IO、数值与发布门禁 |
| **模块七：DeepSeek开放技术详解** | 模型架构、训练与RL、DeepInfer/DSpark、算子通信、3FS/smallpond、集群、视觉与Agent生态、本地复现实验 |
| **面试宝典** | 收录 180+ 场面试真题，覆盖 60+ 家公司，按公司梯队分类 |

每篇文章都遵循「先白话后术语」的写作原则——先用通俗的语言解释"是什么、为什么需要"，再给出严谨的技术细节，确保读者既看得懂也学得对。

<br>

## 🧭 AIInfra 学习路线

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 1 | [AI Infra 学习路线](<docs/guides/AI Infra学习路线.md>) | AI Infra 技术栈全貌、系统化学习路径与知识图谱 |
| 2 | [训练、推理与DeepSeek进阶导航](docs/guides/训练推理与DeepSeek进阶学习导航.md) | 七模块、分岗路线与新增专题的网页入口 |
| 3 | [资料来源、版本与覆盖边界](docs/guides/资料来源版本与覆盖边界.md) | 三份参考材料、固定源码与实验证据的区分 |

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
| 第 1 章 | [编程语言基础](<docs/guides/模块一-前置知识/第1章-编程语言基础.md>) | Python 工程化与并发、C/C++ 内存管理与构建、Python/C++ 互操作、Linux 与 GPU 软件栈 |
| 第 2 章 | [数学基础](<docs/guides/模块一-前置知识/第2章-数学基础.md>) | 线性代数与 GEMM 分块、Softmax/交叉熵/KL 散度、反向传播、数值计算与混合精度 |
| 第 3 章 | [🔥 Transformer 架构：快速入门篇](<docs/guides/模块一-前置知识/transformer/Transformer架构快速入门.md>) | Encoder-Decoder 结构、Self-Attention 机制入门 |
| 3.1 | [AI Infra 工程师为什么必须懂 Transformer](<docs/guides/模块一-前置知识/transformer/3.1 AI Infra工程师为什么必须懂Transformer.md>) | 从 Infra 视角理解 Transformer 的计算与显存特性 |
| 3.2 | [Transformer 全貌及代码实现](<docs/guides/模块一-前置知识/transformer/3.2 Transformer全貌及代码实现.md>) | 完整 Transformer 架构拆解与 PyTorch 实现 |
| 3.3 | [Self-Attention 机制深入理解](<docs/guides/模块一-前置知识/transformer/3.3 Self-Attention机制深入理解.md>) | QKV 与缩放点积、MHA/MQA/GQA/MLA 对比、Causal Mask、计算瓶颈分析 |
| 3.4 | [Transformer 前馈网络 FFN 深入理解](<docs/guides/模块一-前置知识/transformer/3.4 Transformer前馈网络FFN深入理解.md>) | 激活函数与 SwiGLU、中间维度选择、FFN 的张量并行切分、MoE |
| 3.5 | [Transformer 位置编码深入理解](<docs/guides/模块一-前置知识/transformer/3.5 Transformer位置编码深入理解.md>) | Sinusoidal、可学习位置编码、RoPE、ALiBi 与长上下文扩展 |
| 3.6 | [LayerNorm 与残差连接深入理解](<docs/guides/模块一-前置知识/transformer/3.6 LayerNorm与残差连接深入理解.md>) | 残差连接、归一化方法全景、Pre-Norm 与 Post-Norm、CUDA 优化视角 |
| 3.7 | [Transformer Decoder Block 完整解析](<docs/guides/模块一-前置知识/transformer/3.7 Transformer Decoder Block完整解析.md>) | Decoder-only 选型、完整数据流、PyTorch 实现、参数量/FLOPs/显存手算 |
| 3.8 | [从 Transformer 到 LLM 自回归生成深入理解](<docs/guides/模块一-前置知识/transformer/3.8 从Transformer到LLM自回归生成深入理解.md>) | 采样策略、Prefill 与 Decode 阶段、KV Cache 及其优化、推理性能指标 |
| 3.9 | [Tokenization 与词嵌入](<docs/guides/模块一-前置知识/transformer/3.9 Tokenization与词嵌入.md>) | 分词算法、从独热编码到 Word Embedding、Transformer 输入层的构造 |
| 第 4 章 | [🔥 PyTorch 框架快速入门篇](<docs/guides/模块一-前置知识/pyroch/PyTorch框架入门.md>) | Tensor、autograd、nn.Module、完整训练循环、GPU 训练与性能分析入门 |
| 第 5 章 | [🔥 GPU 基础知识：从硬件架构到 AI 计算](<docs/guides/模块一-前置知识/gpu/gpu-basics.md>) | CPU vs GPU、SM 架构、显存层级、Tensor Core |
| 5.1 | [NVIDIA GPU 架构演进：从 Volta 到 Blackwell](<docs/guides/模块一-前置知识/gpu/nvidia-gpu-evolution.md>) | V100 → A100 → H100 → B200 架构演进 |
| 第 6 章 | [🔥 集群通信网络与 NCCL：分布式训练的通信骨架](<docs/guides/模块一-前置知识/communication/collective-communication-primer.md>) | 通信原语、Ring AllReduce 算法、NCCL 实战 |

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
| 快速入门 | [🔥 CUDA 编程快速入门指南](<docs/guides/模块二-CUDA编程与算子优化/CUDA编程入门指南.md>) | CUDA 编程模型、线程层级、内存模型与 Kernel 编写 |
| 第 1 章 | [CUDA 编程入门](<docs/guides/模块二-CUDA编程与算子优化/第1章-CUDA编程入门.md>) | 开发环境搭建、编程模型、内存模型、第一个实用 Kernel |
| 1.1 | [CUDA 开发环境搭建](<docs/guides/模块二-CUDA编程与算子优化/1.1-CUDA开发环境搭建.md>) | 驱动与 Toolkit 安装、nvcc 与 CMake、多版本 CUDA 管理、常见问题排查 |
| 1.2 | [CUDA 编程模型](<docs/guides/模块二-CUDA编程与算子优化/1.2-CUDA编程模型.md>) | Grid/Block/Thread 三级线程层次、线程索引计算、Kernel 启动配置 |
| 1.3 | [CUDA 内存模型](<docs/guides/模块二-CUDA编程与算子优化/1.3-CUDA内存模型.md>) | 全局、共享、寄存器、常量与统一内存的特性、适用场景与选型 |
| 1.4 | [第一个实用 Kernel：向量加法](<docs/guides/模块二-CUDA编程与算子优化/1.4-第一个实用Kernel.md>) | 用向量加法走通 Kernel 编写、主机端代码、错误检查与性能测量 |
| 第 2 章 | [CUDA 性能优化基础](<docs/guides/模块二-CUDA编程与算子优化/第2章-CUDA性能优化基础.md>) | Warp 与执行模型、内存访问优化、Occupancy 与资源分配、同步与原子操作 |
| 2.1 | [Warp 与执行模型](<docs/guides/模块二-CUDA编程与算子优化/2.1-Warp与执行模型.md>) | SIMT 执行模型、Warp Divergence、Warp Shuffle 与 Warp 级归约 |
| 2.2 | [内存访问优化](<docs/guides/模块二-CUDA编程与算子优化/2.2-内存访问优化.md>) | 全局内存合并访问、Shared Memory Bank Conflict、向量化加载 |
| 2.3 | [Occupancy 与资源分配](<docs/guides/模块二-CUDA编程与算子优化/2.3-Occupancy与资源分配.md>) | SM 资源清单、影响 Occupancy 的三大因素、计算实例与调优策略 |
| 2.4 | [同步与原子操作](<docs/guides/模块二-CUDA编程与算子优化/2.4-同步与原子操作.md>) | Block 内与 Warp 级同步、Memory Fence、原子操作优化、Cooperative Groups |
| 第 3 章 | [经典算子实现 - Reduce](<docs/guides/模块二-CUDA编程与算子优化/第3章-经典算子实现-Reduce.md>) | 朴素实现、共享内存+树形归约、Warp Shuffle 优化、多级归约 |
| 3.1 | [CUDA Reduce 算子优化](<docs/guides/模块二-CUDA编程与算子优化/3.1-CUDA Reduce算子优化.md>) | 从朴素规约到 Warp Shuffle 与向量化加载，8 个版本逐步优化 |
| 第 4 章 | [经典算子实现 - GEMM](<docs/guides/模块二-CUDA编程与算子优化/第4章-经典算子实现-GEMM.md>) | 矩阵乘法基础、Shared Memory Tiling、进一步优化、与 cuBLAS 对比 |
| 4.1 | [CUDA GEMM 算子性能优化](<docs/guides/模块二-CUDA编程与算子优化/4.1-CUDA GEMM算子性能优化.md>) | Block/Warp/Thread 三级 Tiling、float4 向量化、Bank Conflict 消除、双缓冲与异步拷贝 |
| 第 5 章 | [经典算子实现 - Softmax 与算子融合](<docs/guides/模块二-CUDA编程与算子优化/第5章-经典算子实现-Softmax与算子融合.md>) | Softmax 数值稳定实现、Online Softmax、算子融合 |
| 5.1 | [CUDA Softmax 朴素实现优化](<docs/guides/模块二-CUDA编程与算子优化/5.1-CUDA Softmax朴素实现优化.md>) | Safe Softmax、Warp Shuffle 规约、向量化加载、两遍融合 Kernel |
| 5.2 | [CUDA Online Softmax 实现优化](<docs/guides/模块二-CUDA编程与算子优化/5.2-CUDA Online Softmax实现.md>) | Online Softmax 推导与 CUDA 实现，FlashAttention 的前置知识 |
| 第 6 章 | [Attention 算子](<docs/guides/模块二-CUDA编程与算子优化/第6章-Attention算子.md>) | FlashAttention V1/V2/V3、Decode 阶段加速、PagedAttention CUDA 实现 |
| 6.1 | [FlashAttention V1 详解](<docs/guides/模块二-CUDA编程与算子优化/6.1-FlashAttention V1详解.md>) | Tiling + Online Softmax 把额外显存从 $O(N^2)$ 降到 $O(N)$，IO 复杂度分析与反向重计算 |
| 6.2 | [FlashAttention V2 详解](<docs/guides/模块二-CUDA编程与算子优化/6.2-FlashAttention V2详解.md>) | 调换内外循环、减少非矩阵乘运算、Warp 级工作分配、Causal Mask 优化 |
| 第 7 章 | [AI 编译器](<docs/guides/模块二-CUDA编程与算子优化/第7章-AI编译器.md>) | Triton、torch.compile、TVM/XLA 概述 |
| 7.1 | [Triton 编程模型](<docs/guides/模块二-CUDA编程与算子优化/7.1-Triton编程框架.md>) | Triton 编程模型与基础语法，以 Softmax 为例与 PyTorch 实现对比性能 |


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
| 第 1 章 | [分布式训练总论](<docs/guides/模块三-分布式训练/第1章-分布式训练总论.md>) | 为什么需要分布式训练、训练显存账本、五大并行策略全景、环境搭建与 DDP 启动 |
| 1.1 | [分布式训练总论：显存账本与五大并行策略全景](<docs/guides/模块三-分布式训练/1.1 分布式训练总论.md>) | 手算训练显存账本，总览五大并行策略全景与选择原则 |
| 1.2 | [环境搭建与分布式启动](<docs/guides/模块三-分布式训练/1.2 环境搭建与分布式启动.md>) | rank/local_rank/world_size、torchrun 启动与会合机制、NCCL 调试、最小 DDP 脚本 |
| 第 2 章 | [集合通信原语](<docs/guides/模块三-分布式训练/第2章-集合通信原语.md>) | AllReduce/ReduceScatter/AllGather/All-to-All 语义、Ring 算法、通信量量化分析 |
| 2.1 | [集合通信原语详解](<docs/guides/模块三-分布式训练/2.1 集合通信原语详解.md>) | AllReduce/ReduceScatter/AllGather/All-to-All 语义、Ring AllReduce 原理、通信耗时量化 |
| 第 3 章 | [优化器](<docs/guides/模块三-分布式训练/第3章-优化器.md>) | SGD/Adam/AdamW 演进、优化器状态显存开销分析、大 Batch 优化器 LAMB/LARS |
| 3.1 | [优化器：从 SGD 到 AdamW 与显存开销分析](<docs/guides/模块三-分布式训练/3.1 优化器原理与显存开销分析.md>) | 优化器演进、优化器状态显存开销、FP32 主权重的必要性、LARS/LAMB |
| 第 4 章 | [数据并行](<docs/guides/模块三-分布式训练/第4章-数据并行.md>) | DataParallel、DistributedDataParallel、FSDP |
| 4.1 | [数据并行详解](<docs/guides/模块三-分布式训练/4.1 数据并行详解.md>) | DP/DDP/FSDP 演进、梯度同步机制、通信量与显存账本的数学推导 |
| 4.2 | [PyTorch 数据并行从原理到实战](<docs/guides/模块三-分布式训练/4.2 PyTorch 数据并行从原理到实战.md>) | 通信原语、DDP、FSDP 参数分片、多机多卡实战，附完整代码示例 |

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

### 新增训练主线（2026-11-07）

| 阅读入口 | 新增内容 |
|---|---|
| [第12章：训练数据系统](docs/guides/模块三-分布式训练/第12章-训练数据系统.md) | 治理去重、mixture、tokenizer、GPTDataset三索引、packing与存储吞吐 |
| [第13章：Megatron源码与训练状态](docs/guides/模块三-分布式训练/第13章-Megatron源码与训练状态.md) | 迭代、ModuleSpec/TE、main_grad/DistOpt、Hybrid/GTP、MLA/MTP与PGCollection |
| [第14章：大规模训练可靠性](docs/guides/模块三-分布式训练/第14章-大规模训练可靠性.md) | checkpoint staging/finalize、重分片、RNG、fail-slow/SDC、调度与故障演练 |
| [第15章：超大规模训练设计与成本](docs/guides/模块三-分布式训练/第15章-超大规模训练设计与成本.md) | 参数/FLOPs、五类配置、训练课程、optimizer、scaling laws与μP |
| [第16章：训练源码题库与项目](docs/guides/模块三-分布式训练/第16章-训练源码题库与项目实战.md) | 40道完整问答、8个项目合同、12周路线 |
| [第17章：训练框架与异构生态](docs/guides/模块三-分布式训练/第17章-训练框架与异构加速生态.md) | TorchTitan、MaxText、开放框架与AMD/TPU/Trainium迁移合约 |
| [HSDP与多实例DistOpt](docs/guides/模块三-分布式训练/5.3-HSDP与多实例分布式优化器.md) | inner/outer状态、两级归约与d=D/I |
| [FP8/FP4缩放](docs/guides/模块三-分布式训练/8.3-FP8与FP4训练的缩放数值与验收.md) | amax、块缩放、转置、BF16 shadow与验收 |
| [长上下文](docs/guides/模块三-分布式训练/第9章-长序列训练与上下文并行.md) | RoPE课程、混合长度、有效评测、AutoSP与HexiSeq |

训练覆盖按机制与来源段落复核，详见[训练覆盖审计](docs/expansion-2026/TRAINING_COVERAGE_AUDIT.md)。40道代表题和项目规格不等于原Megatron报告全部126题逐项实现。

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
| 第 1 章 | [LLM 推理基础](<docs/guides/模块四-推理优化/第1章-LLM推理基础/第1章-LLM推理基础.md>) | Prefill/Decode 两阶段、KV Cache、性能指标、Roofline 瓶颈分析、推理链路 |
| 1.1 | [LLM 推理基础](<docs/guides/模块四-推理优化/第1章-LLM推理基础/1.1-LLM推理基础.md>) | Prefill/Decode 两阶段、KV Cache 显存账本、TTFT/TPOT 指标，用 Roofline 解释 Decode 为何 Memory Bound |
| 1.2 | [vLLM 快速入门](<docs/guides/模块四-推理优化/第1章-LLM推理基础/1.2-vLLM快速入门.md>) | 从安装到部署你的第一个 LLM 推理服务，离线批量推理与在线 OpenAI 兼容服务 |
| 第 2 章 | [推理引擎核心技术](<docs/guides/模块四-推理优化/第2章-推理引擎核心技术/第2章-推理引擎核心技术.md>) | PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill、Attention 后端与图优化 |
| 2.1 | [PagedAttention](<docs/guides/模块四-推理优化/第2章-推理引擎核心技术/2.1-PagedAttention.md>) | 借鉴虚拟内存分页思想，用 Block Table 消除 KV Cache 碎片化，支持前缀共享与 Copy-on-Write |
| 2.2 | [Continuous Batching](<docs/guides/模块四-推理优化/第2章-推理引擎核心技术/2.2-Continuous Batching.md>) | Iteration-level Scheduling 让请求随到随拼、完成即退，减少固定 batch 的尾部空转 |
| 2.3 | [Prefix Cache 与 RadixAttention](<docs/guides/模块四-推理优化/第2章-推理引擎核心技术/2.3-Prefix Cache 与 RadixAttention.md>) | vLLM 基于 Hash 的自动前缀缓存，SGLang 用 Radix Tree 做更高效的前缀共享 |
| 2.4 | [Chunked Prefill 与统一调度](<docs/guides/模块四-推理优化/第2章-推理引擎核心技术/2.4-Chunked Prefill 与统一调度.md>) | 切块 Prefill 消除对 Decode 的干扰，vLLM V1 用统一 Token 预算调度器抹平 Prefill/Decode 边界 |
| 2.5 | [Attention 后端与图优化](<docs/guides/模块四-推理优化/第2章-推理引擎核心技术/2.5-Attention 后端与图优化.md>) | 可插拔 Attention 后端与 CUDA Graph、torch.compile 消除 Decode 阶段的 CPU 启动开销 |
| 第 3 章 | [nano-vllm 源码解读](<docs/guides/模块四-推理优化/第3章-nano-vllm源码解读/第3章-nano-vllm源码解读.md>) | 以约 1200 行极简引擎 nano-vllm 为主线，逐模块读懂连续批处理、PagedAttention、FlashAttention、张量并行与 CUDA Graph |
| 3.1 | [项目全景与四层架构](<docs/guides/模块四-推理优化/第3章-nano-vllm源码解读/3.1-项目全景与四层架构.md>) | nano-vllm 定位、目录结构，与接口/引擎/调度/执行四层架构及端到端数据流 |
| 3.2 | [配置、入口与一次 generate 的全流程](<docs/guides/模块四-推理优化/第3章-nano-vllm源码解读/3.2-配置入口与推理主循环.md>) | Config/SamplingParams、LLM 对 LLMEngine 的封装，以及 generate→add_request→step→postprocess 主循环 |
| 3.3 | [模型骨架与核心算子](<docs/guides/模块四-推理优化/第3章-nano-vllm源码解读/3.3-模型与核心算子.md>) | RMSNorm/RoPE/SwiGLU、并行 Linear 族与张量并行、词表并行 Embedding/LMHead、Attention 两条路径 |
| 3.4 | [KV Cache 与 BlockManager](<docs/guides/模块四-推理优化/第3章-nano-vllm源码解读/3.4-KVCache与BlockManager.md>) | Block 结构、PagedAttention 分块、基于 xxhash 的前缀缓存、块的分配回收与 Sequence 状态机 |
| 3.5 | [Scheduler 与连续批处理](<docs/guides/模块四-推理优化/第3章-nano-vllm源码解读/3.5-Scheduler与连续批处理.md>) | Waiting/Running 双队列、Prefill/Decode 分离调度、Token Budget、chunked prefill 与抢占重算 |
| 3.6 | [ModelRunner、CUDA Graph 与性能基准](<docs/guides/模块四-推理优化/第3章-nano-vllm源码解读/3.6-ModelRunner引擎循环与性能.md>) | 序列张量化、全局 Context、KV Cache 显存切分、CUDA Graph 捕获回放，以及官方基准与 vLLM 对比 |
| 第 4 章 | [量化](<docs/guides/模块四-推理优化/第4章-量化/第4章-量化.md>) | 量化基础、SmoothQuant、GPTQ/AWQ、FP8、NVFP4/MXFP4、KV Cache 量化 |
| 4.1 | [量化基础](<docs/guides/模块四-推理优化/第4章-量化/4.1-量化基础.md>) | Scale 与零点、对称与非对称、位宽与格距、量化粒度、校准与 PTQ/QAT |
| 4.2 | [W8A8 与 SmoothQuant](<docs/guides/模块四-推理优化/第4章-量化/4.2-W8A8与SmoothQuant.md>) | 激活离群值、等价缩放与迁移强度 α、INT8 GEMM、LLM Compressor 配方 |
| 4.3 | [Weight-only INT4](<docs/guides/模块四-推理优化/第4章-量化/4.3-Weight-only-INT4.md>) | GPTQ 误差补偿、AWQ 激活感知缩放、Marlin 融合 Kernel 与 vLLM 的 Kernel 选择 |
| 4.4 | [KV Cache 量化](<docs/guides/模块四-推理优化/第4章-量化/4.4-KV-Cache量化.md>) | KIVI 分组与残留缓存、FP8 KV 的 Scale 校准与跳层、长上下文验证 |
| 4.5 | [FP8 与 NVFP4/MXFP4](<docs/guides/模块四-推理优化/第4章-量化/4.5-FP8与NVFP4-MXFP4.md>) | E4M3/E5M2、FP8 Scale、E2M1 微块缩放、NVFP4 与 MXFP4 对比及硬件执行条件 |
| 4.6 | [量化选型与 vLLM 实战](<docs/guides/模块四-推理优化/第4章-量化/4.6-量化选型与vLLM实战.md>) | 基于 v0.32.0 的按瓶颈选型、在线量化入口、FP16/AWQ 吞吐与生成质量对照 |
| 第 5 章 | [Speculative Decoding](<docs/guides/模块四-推理优化/第5章-Speculative-Decoding/第5章-Speculative-Decoding.md>) | Draft+Verify 原理、N-gram/Suffix、Medusa/EAGLE/MTP、vLLM 投机解码实战 |
| 5.1 | [投机解码核心原理](<docs/guides/模块四-推理优化/第5章-Speculative-Decoding/5.1-投机解码核心原理.md>) | Draft + Verify、接受与残差采样的保分布证明、one-hot Draft、KV 回退 |
| 5.2 | [Draft 模型与无模型方案](<docs/guides/模块四-推理优化/第5章-Speculative-Decoding/5.2-Draft模型与无模型方案.md>) | 独立 Draft 与 PARD、Tokenizer 兼容、N-gram/Suffix、接受率与候选覆盖 |
| 5.3 | [Medusa、EAGLE 与 MTP](<docs/guides/模块四-推理优化/第5章-Speculative-Decoding/5.3-Medusa与EAGLE.md>) | 多头预测、Tree Attention、EAGLE-2/3、DeepSeek-V3 MTP、Block Verification |
| 5.4 | [收益边界与限制](<docs/guides/模块四-推理优化/第5章-Speculative-Decoding/5.4-收益边界与限制.md>) | 提交长度与轮耗时账本、投机长度递减、Dynamic SD、并发、量化与流式延迟 |
| 5.5 | [vLLM 投机解码实战](<docs/guides/模块四-推理优化/第5章-Speculative-Decoding/5.5-vLLM投机解码实战.md>) | 基于 v0.32.0 的基线/N-gram/Draft/EAGLE-3 对照、逐请求接受指标与 Benchmark |
| 第 10 章 | [端侧推理](<docs/guides/模块四-推理优化/第10章-端侧推理/第10章-端侧推理.md>) | 端侧软硬件栈、异构执行、运行时选型、量化、内存优化与 Benchmark |
| 10.1 | [端侧推理基础：从模型导出到异构硬件执行](<docs/guides/模块四-推理优化/第10章-端侧推理/10.1-端侧推理基础.md>) | 端侧约束、软硬件栈、运行时选型、量化与内存优化、Benchmark，以及 ExecuTorch/XNNPACK 最小实例 |

<br>

### 新增源码、生产与题库入口

- [第6—9章工程主线](docs/guides/模块四-推理优化/第6章-分布式推理.md)：分布式容量、WideEP、PD/KV传输、Benchmark、流式背压与故障演练。
- [第11章：SGLang与vLLM源码数据流](docs/guides/模块四-推理优化/第11章-SGLang与vLLM源码数据流.md)：请求、Scheduler、KV、ModelRunner及结构化输出/投机状态。
- [第12章：源码题库与项目](docs/guides/模块四-推理优化/第12章-推理源码题库与项目实战.md)：计算题、系统设计、项目与分岗模拟面试。
- [235题逐题速查](docs/guides/模块四-推理优化/12.3-推理面试235题索引与逐题速查.md)：保留全部题号、题干、分类和顺序，每题有重写要点与教程入口；不等同于235个完整实现或硬件实验。

来源与逐题映射见[推理覆盖审计](docs/expansion-2026/INFERENCE_COVERAGE_AUDIT.md)。

## 🧠 模块五：LLM RL 后训练

沿用分布式训练模块的「章节导读 + 专题详解」结构，继续展开到长上下文与异步RL工程。每篇结合直觉、公式、数值或系统算例、工程排查、自测答案和一手资料。

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
| [第11章：长上下文与异步RL工程](docs/guides/模块五-LLM-RL后训练/第11章-长上下文与异步RL工程.md) | 工具轨迹Mask、CP/DP loss、GSPO序列ratio、取消/超时/reward版本与环境吞吐 |

配套 [CPU 数值实验](labs/llm_rl/README.md) 无需下载模型或安装 GPU 依赖：

```bash
npm run test:rl
npm run lab:rl
```

实验使用四上下文、四动作策略表，验证数学与状态流转；完整 GPU 框架训练和多机性能未作为已验证结果。

## 📊 模块六：性能工程与生产实践

用同一套方法分析训练与推理：固定工作量，测量时间线，建立资源模型，再用数值和质量护栏验证优化。

| 章节 | 网页教程 |
|---|---|
| 1 性能模型与实验方法 | [Roofline与关键路径](docs/guides/模块六-性能工程与生产实践/1.1-跨训练推理的Roofline与关键路径.md)、[统计测量与归因](docs/guides/模块六-性能工程与生产实践/1.2-统计测量基准设计与性能归因.md) |
| 2 GPU算子编译与调优 | [微架构瓶颈](docs/guides/模块六-性能工程与生产实践/2.1-从GPU时间线到微架构瓶颈.md)、[编译/Graph/动态Shape](docs/guides/模块六-性能工程与生产实践/2.2-编译CUDA图与动态Shape工程.md) |
| 3 通信存储与主机协同 | [NCCL与RDMA](docs/guides/模块六-性能工程与生产实践/3.1-NCCL与RDMA网络性能工程.md)、[NUMA与异步IO](docs/guides/模块六-性能工程与生产实践/3.2-NUMA数据加载与异步IO流水.md) |
| 4 正确性与优化项目 | [低精度Kernel门禁](docs/guides/模块六-性能工程与生产实践/4.1-低精度Kernel正确性与发布门禁.md)、[综合实验](docs/guides/模块六-性能工程与生产实践/4.2-AIInfra性能项目与综合实验.md) |

## 🐋 模块七：DeepSeek 开放技术详解

按模型、训练、推理、算子、存储与生态分层阅读。每篇区分论文报告、开放权重、公开源码、独立oracle和未验证的部署条件。

| 章节 | 阅读入口 |
|---|---|
| 1 版图与证据 | [开放技术版图与版本](docs/guides/模块七-DeepSeek开放技术详解/第1章-开放技术版图与版本证据.md) |
| 2 架构与长上下文 | [MLA/MoE/稀疏Attention/mHC/CED/Engram](docs/guides/模块七-DeepSeek开放技术详解/第2章-模型架构与长上下文.md) |
| 3 预训练 | [FP8/DualPipe/Muon/MTP与DSpark目标](docs/guides/模块七-DeepSeek开放技术详解/第3章-预训练数值与并行系统.md) |
| 4 后训练 | [GRPO/R1/Agent/数学/蒸馏](docs/guides/模块七-DeepSeek开放技术详解/第4章-RL推理与Agent后训练.md) |
| 5 推理执行 | [FlashMLA/DeepInfer/DSpark](docs/guides/模块七-DeepSeek开放技术详解/第5章-推理算子与DeepInfer运行时.md) |
| 6 算子与通信 | [DeepGEMM/DeepEP/EPLB/DeepSelect/DeepJIT/Ascend](docs/guides/模块七-DeepSeek开放技术详解/第6章-算子通信与跨硬件工程.md) |
| 7 存储与集群 | [3FS/smallpond/FireFlyer/DSec](docs/guides/模块七-DeepSeek开放技术详解/第7章-存储数据与集群基础设施.md) |
| 8 多模态与Agent生态 | [VL/Janus/OCR/Harness/Recipe](docs/guides/模块七-DeepSeek开放技术详解/第8章-多模态与Agent开放生态.md) |
| 9 实验与复现 | [本地数学、状态与MLA实验](docs/guides/模块七-DeepSeek开放技术详解/第9章-本地实验与复现验收.md) |

配套[实验说明](labs/deepseek/README.md)区分CPU数学、官方EPLB CPU规划和普通PyTorch MLA GPU验证；没有把它们称为官方kernel或完整模型性能复现。

## 💻 本地浏览

**教程正文都通过本地网站阅读；源码和实验在终端运行。** 浏览器会展示目录、正文、公式、图示、代码示例和自测答案，不会自动执行代码、启动模型或安装CUDA依赖。

准备Node.js与npm，在仓库根目录运行：

```bash
npm ci
npm run site:local
```

构建完成后打开[本地首页](http://localhost:4321/AIInfraGuide/)。七个模块分别位于：

| 模块 | 本地入口 |
|---|---|
| 前置知识 | [prerequisites](http://localhost:4321/AIInfraGuide/prerequisites/) |
| CUDA与算子 | [cuda](http://localhost:4321/AIInfraGuide/cuda/) |
| 分布式训练 | [distributed](http://localhost:4321/AIInfraGuide/distributed/) |
| 推理优化 | [inference](http://localhost:4321/AIInfraGuide/inference/) |
| LLM RL | [rl](http://localhost:4321/AIInfraGuide/rl/) |
| 性能工程 | [performance](http://localhost:4321/AIInfraGuide/performance/) |
| DeepSeek | [deepseek](http://localhost:4321/AIInfraGuide/deepseek/) |

WSL启动后可从Windows浏览器访问同一localhost地址。端口以终端实际输出为准；URL保留`/AIInfraGuide/`前缀。`Ctrl+C`停止服务。

日常编辑使用`npm run dev:local`。搜索索引由构建生成，修改后需重新运行`npm run build`；预览模式也需重新构建才能看到更新。

构建后的内容检查入口是`npm run check:content`。实验命令及Python/PyTorch条件分别见[分布式labs](labs/distributed/README.md)、[RL labs](labs/llm_rl/README.md)、[DeepSeek labs](labs/deepseek/README.md)。外部论文和GitHub链接仍跳转到对应来源。

### 来源与覆盖记录

三份参考分别为训练Infra综述（2026-08-24）、Megatron源码报告（2026-08-29）和推理工程师宝典（2026-08-30）。输入文件SHA与章节映射见[参考清单](docs/expansion-2026/reference-manifest.json)；主要章节按主题重组，未笼统声称全部研究方法、每个参考问题或所有硬件路径已完整实现。

- [网页阅读版：资料来源与覆盖边界](docs/guides/资料来源版本与覆盖边界.md)。
- [训练覆盖审计](docs/expansion-2026/TRAINING_COVERAGE_AUDIT.md)与[推理覆盖审计](docs/expansion-2026/INFERENCE_COVERAGE_AUDIT.md)。
- [工作包与交付记录](docs/expansion-2026/PLAN.md)与[文章更新日志](docs/posts/update-log.md)。

## 🎯 面试宝典

共收录 **180+ 场面试真题**，覆盖 **60+ 家公司**，按梯队分类组织。[在线浏览 →](https://caomaolufei.github.io/AIInfraGuide/interview)

涵盖公司包括字节跳动、阿里巴巴、腾讯、百度、快手、美团、蚂蚁、英伟达、MiniMax、蔚来、小鹏、理想等。

<img src="https://caomaolufei.github.io/AIInfraGuide/images/AIinfraGuideWeb2.png" alt="AIinfraGuideWeb2" style="max-width: 100%; display: block; margin: 0 auto;" />

<br>

**投递总览：**

| 梯队 | 公司 | 主要 AI Infra 岗位 |
|------|------|------|
| **T0 大厂** | 字节跳动、阿里巴巴、腾讯、百度 | AI Infra 工程师、高性能计算研发、推理优化工程师、**分布式训练框架** |
| **T1 大厂/独角兽** | 快手、小米、美团、蚂蚁、京东、拼多多、OPPO、华为 | AI Infra、高性能计算、大模型推理优化、AI 平台开发 |
| **T2 AI 独角兽** | MiniMax、阶跃星辰、智谱AI、面壁智能、月之暗面 | AI Infra、大模型算法 (偏 Infra)、推理系统开发、Agent Infra |
| **T3 芯片/硬件厂商** | 英伟达、摩尔线程、海光、寒武纪、壁仞科技、飞腾 | 算子开发、CUDA 优化、GPU 软件工程师、AI 编译器、高性能计算 |
| **T4 自动驾驶/车企** | 小鹏汽车、蔚来、理想、卓驭(大疆车载)、小马智行、元戎启行 | AI Infra、高性能计算、大模型推理优化、AI 平台 |
| **T5 其他** | 科大讯飞、网易、海康威视、联想、猿辅导、好未来等 | AI Infra、高性能计算、推理引擎开发、大模型算法 |

> 学习侧重点按目标岗位选择：训练岗位重点掌握数据、并行、更新与恢复；Serving岗位重点掌握请求、KV、调度和SLO；算子岗位重点掌握硬件、数值与性能证据。可使用进阶导航与分岗模拟路线安排练习。

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

- 公众号：[《AI Infra Guide》](https://mp.weixin.qq.com/s/oxUBDTd-b1aZnh0jGR5tqw)
- 知乎：[草帽路飞](https://zhihu.com/people/zi-you-zi-zai-de-zhan-sheng)
- 社群：微信 caomaolufei2026

<img src="https://caomaolufei.github.io/AIInfraGuide/images/wx_qun.png" alt="wx_qun" style="max-width: 50%; display: block; margin: 0 auto;" />

## License

MIT
