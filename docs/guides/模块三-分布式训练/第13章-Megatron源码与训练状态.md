---
title: "第13章：Megatron源码与训练状态"
description: "沿训练迭代、模块构造、梯度与参数同步追踪状态所有权，并区分冻结源码与版本化新能力"
pubDate: 2026-11-07
category: "distributed-training"
order: 130
tags: ["Megatron源码", "状态生命周期", "可复现性"]
---

## 本章解决的问题

读训练框架最容易迷路的原因，是把“谁调用谁”与“谁拥有状态”分开研究。

一个 Python 函数返回了，并不说明它发出的 GPU 通信已经完成；一个参数有完整 shape，也不说明它的更新状态由当前 rank 全量持有。

本章把调用链和状态表放在一起。每到一个边界都检查输入、输出、所有者以及下一位消费者。

这样可以从一个错误参数反推到数据、反向、归约、优化器或参数同步阶段，而不用先读完整个仓库。

## 状态与不变量

| 阶段 | 必须记录的对象 | 验收问题 |
|---|---|---|
| 初始化 | rank、process group、RNG | 所有参与者使用同一协议 |
| 模型构造 | spec、global shape、local shard | 并行布局与后端合法 |
| 调度 | microbatch、activation、P2P | 同一序列的前后向匹配 |
| 梯度 | main_grad、ready count、通信句柄 | 未完成不能消费 |
| 更新 | master、Adam moments、局部范围 | 同一逻辑坐标一起更新 |
| 下一前向 | 参数buffer、AG完成事件 | 不读旧或局部参数 |

每一行都需要明确读写边界。

只记录最终配置，无法解释训练过程中的状态变化。

只记录运行日志，也无法在底层数据或代码发生变化后重新构造同一任务。

因此应保存不可变版本、运行状态和观测证据三类信息。

## 专题导航

### 13.1 一次Megatron训练迭代的完整调用链

从pretrain_gpt的回调走入初始化、schedule、loss closure、梯度归约和optimizer提交。

阅读时先完成文中的手算，再检查源码所要求的前置条件。

每篇都包含故障定位、自测答案和官方资料入口。

### 13.2 ModuleSpec模型构造与TransformerEngine

从模块语法树追到QKV、SwiGLU、tuple contract及TE后端与硬件门槛。

阅读时先完成文中的手算，再检查源码所要求的前置条件。

每篇都包含故障定位、自测答案和官方资料入口。

### 13.3 梯度Buffer分布式优化器与状态所有权

沿一个参数区间追踪main_grad、bucket、Reduce-Scatter、本地Adam更新和参数All-Gather。

阅读时先完成文中的手算，再检查源码所要求的前置条件。

每篇都包含故障定位、自测答案和官方资料入口。

### 13.4 HybridModel与广义张量并行

解释GPT block拆成positions对checkpoint和PP的影响，再用显存与通信算例理解GTP。

阅读时先完成文中的手算，再检查源码所要求的前置条件。

每篇都包含故障定位、自测答案和官方资料入口。

### 13.5 Megatron中MLA与MTP的源码Gate

从普通MLA与latent-cache路径走到MTP的projection、D+1 hidden、有效token、CP roll和detach边界。

用8 token zigzag与packed文档反例核验跨rank标签移位，并同时保留0.19文档与tag源码的冲突。

### 13.6 ProcessGroupCollection与动态混合CP

把group作为组件依赖，核对required字段、legacy适配和多模块grid，再手算32卡dense/expert双视图。

用2幂资源公式解释动态hybrid CP，连接工作量、group切换barrier、末subsample同步和配置gate。

## 建议的学习顺序

13.1 先读主调用链，不在第一次遇到某函数时无限展开。先标注它接收什么状态、承诺什么完成条件。

13.2 选择一个小尺寸 GQA+SwiGLU 层，手算每个 local shape 后再对照 spec 插槽。

13.3 用一个 12 元素 bucket 和 3 个 DP rank 理解参数可以跨 optimizer shard 边界。

13.4 单独讨论 0.19.0 文档描述的 HybridModel 与 GTP，不把新能力塞回旧 commit 的调用链。

13.5 对照具体tensor理解MLA/MTP：先做单rank数值与目标对齐，再加SP、CP和PP。

13.6 从显式PG依赖检查同一模型的不同rank视图，最后研究动态group和调度边界。

## 统一的证据口径

本章由两份既有学习资料扩写，而不是把资料日期当作新验证日期。

第一份是《LLM超大规模与超长上下文训练Infra技术综述_2026》，资料快照日期为2026-08-24。

第二份是《Megatron-LM-20260829/ANALYSIS_REPORT》，资料冻结日期为2026-08-29。

本轮查阅日期为2026-11-07。

其源码主线固定于Megatron提交`48a887fecd01674346f724dcf44f417f455989f0`。

本地Megatron工作树当前HEAD为`2052769557dcf7045c4091d1996f0bc59afeb80e`，与该冻结提交不同。

源码复核使用`git show 48a887f:路径`，不会借用当前工作树行号冒充旧版本证据。

涉及HybridModel与GTP的新增合约，另引NVIDIA的`0.19.0`版本化文档。

这不构成“0.19.0是2026-11-07最新版本”的声明。

| 证据 | 本章允许的结论 |
|---|---|
| 数学推导 | 在列出的假设与单位下成立 |
| 小型Python示例 | 展示索引、范围、状态或计数算法 |
| 固定commit源码 | 说明该版本的调用、判断和数据布局 |
| 版本化官方文档 | 说明特定release的功能合约 |
| 论文案例 | 说明论文报告环境中的实验结果 |
| 本章没有执行的多卡训练 | 不声称性能、收敛或生产可用性已实测 |

## 怎样阅读源码链接

源码链接固定commit，因此可以与文字描述互相核对。

优先从入口、状态结构和消费者三处交叉阅读。

函数名相同并不保证不同版本拥有相同实现。

如果使用其他commit，应重新检查配置校验、依赖版本和对应测试。

文档中的示意代码会明确标注简化部分。

它们不替代Megatron启动器、真实数据预处理或生产存储协议。

## 学完应能完成的任务

- 用一次 global step 解释 forward_step 为什么会被多次调用。
- 从 local tensor shape 推导需要哪种 collective。
- 说明普通 DDP 与 DistOpt 的param storage行为不同。
- 解释 fused wgrad 已写main_grad时为何不能再无条件add。
- 列出RS完成、本地更新完成、AG完成三个不同消费边界。
- 区分模型迁移、optimizer迁移、数据/RNG连续性与性能验收。

## 从教程到实验的三步

1. 在纸上完成文中的小例子，写出期望输出。
2. 用CPU算法或固定shape检查验证索引、坐标和计数。
3. 在目标硬件与依赖下补全分布式正确性、性能和故障实验。

第三步应记录commit、容器、依赖、GPU、互联和数据版本。

增加GPU数量之前，先证明单步状态与恢复逻辑一致。

性能测量应有预热、计时边界、有效工作量和长尾分布。

任何跨拓扑变化都应重新检查数据映射、RNG与归约顺序。

## 本章交付物建议

- 一份带版本的配置与状态清单。
- 一组可以手算的小输入和期望输出。
- 一张源码入口到消费者的调用图。
- 一份按阶段保存的计数、摘要或时间戳。
- 一份正常路径与失败路径的验收记录。

这些产物帮助判断一次改动改变的是算法、布局、运行时还是平台。

它们也让恢复和迁移的限制可以被复核，而不依赖口头经验。

## 参考入口

- [Megatron固定源码](https://github.com/NVIDIA/Megatron-LM/tree/48a887fecd01674346f724dcf44f417f455989f0)。
- [Megatron Core 0.19.0](https://docs.nvidia.com/megatron-core/developer-guide/0.19.0/)：只按版本化合约使用。
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)：结合所安装PyTorch版本核对API。

专题正文给出更细的来源小节映射与文件级证据。
