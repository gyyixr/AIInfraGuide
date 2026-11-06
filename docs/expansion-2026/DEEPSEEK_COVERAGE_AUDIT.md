# DeepSeek 开放技术覆盖审计

核对日期：2026-11-07。该审计覆盖本次新增模块的九个章节；技术依据为正文所引官方论文、官方仓库与模型卡的固定快照。文件数只用于交付盘点，机制是否说明以正文和以下核对记录为准。

## 1. 章节与正文清单

| 章 | 深度核对要点 | 详解入口 |
|---|---|---|
| 1 | 模型/代码/权重/报告分栏；历史 LLM、Coder/FIM、ESFT 到当前固定版本，不以项目存在推断全部训练系统开放。 | [1.1](../guides/模块七-DeepSeek开放技术详解/1.1-DeepSeek模型与开源项目演进.md)、[1.2](../guides/模块七-DeepSeek开放技术详解/1.2-开源代码开放权重与论文披露边界.md)、[1.3](../guides/模块七-DeepSeek开放技术详解/1.3-DeepSeekLLM-Coder与ESFT的早期技术路线.md) |
| 2 | MLA 矩阵吸收与 RoPE、MoE 路由、稀疏注意力、mHC、CED、Engram、CSA2/HSI：逐项给 shape、数值或状态边界。 | [2.1](../guides/模块七-DeepSeek开放技术详解/2.1-MLA低秩压缩解耦RoPE与矩阵吸收.md)、[2.2](../guides/模块七-DeepSeek开放技术详解/2.2-DeepSeekMoE细粒度专家与共享专家.md)、[2.3](../guides/模块七-DeepSeek开放技术详解/2.3-NSA与DSA稀疏注意力.md)、[2.4](../guides/模块七-DeepSeek开放技术详解/2.4-V4混合注意力CSA与HCA.md)、[2.5](../guides/模块七-DeepSeek开放技术详解/2.5-mHC残差流形约束与数值稳定性.md)、[2.6](../guides/模块七-DeepSeek开放技术详解/2.6-V4点1架构更新与原生视觉.md)、[2.7](../guides/模块七-DeepSeek开放技术详解/2.7-CED编解码解耦与PrefixReplay.md)、[2.8](../guides/模块七-DeepSeek开放技术详解/2.8-Engram条件记忆与分层卸载.md)、[2.9](../guides/模块七-DeepSeek开放技术详解/2.9-CSA2与分层稀疏索引.md) |
| 3 | FP8 scale 与累积、DualPipe 时序、Muon 数值、MTP/DSpark 训练目标；框架示例和完整主模型训练配方分开。 | [3.1](../guides/模块七-DeepSeek开放技术详解/3.1-FP8细粒度缩放与训练数值链路.md)、[3.2](../guides/模块七-DeepSeek开放技术详解/3.2-DualPipe与DualPipeV流水调度.md)、[3.3](../guides/模块七-DeepSeek开放技术详解/3.3-Muon优化器与V4训练配方.md)、[3.4](../guides/模块七-DeepSeek开放技术详解/3.4-MTP与DSpark训练目标.md) |
| 4 | R1/GRPO 路线、版本化优势/KL 定义、Agent 环境、数学自验证、蒸馏与反馈；有模型权重不等于公开全部奖励/数据。 | [4.1](../guides/模块七-DeepSeek开放技术详解/4.1-GRPO与R1Zero到R1的训练路线.md)、[4.2](../guides/模块七-DeepSeek开放技术详解/4.2-V3点2与V4点1AgentRL和环境系统.md)、[4.3](../guides/模块七-DeepSeek开放技术详解/4.3-DeepSeekMath与自验证推理.md)、[4.4](../guides/模块七-DeepSeek开放技术详解/4.4-蒸馏合成数据与1017自反馈更新.md) |
| 5 | FlashMLA 物理索引/缓存格式、DeepInfer 请求与输出生命周期、DSpark 非预知决策；CPU oracle 与目标硬件 kernel 分开。 | [5.1](../guides/模块七-DeepSeek开放技术详解/5.1-FlashMLA稀疏算子与KV缓存格式.md)、[5.2](../guides/模块七-DeepSeek开放技术详解/5.2-DeepInfer端到端源码与执行生命周期.md)、[5.3](../guides/模块七-DeepSeek开放技术详解/5.3-DSpark投机验证与置信度调度.md) |
| 6 | 低精度 GEMM、EP 计数/事件、专家副本、TopK、JIT 缓存、Ascend API 能力；固定版本和设备条件逐项说明。 | [6.1](../guides/模块七-DeepSeek开放技术详解/6.1-DeepGEMM低精度矩阵与MegaMoE.md)、[6.2](../guides/模块七-DeepSeek开放技术详解/6.2-DeepEP通信布局与异步生命周期.md)、[6.3](../guides/模块七-DeepSeek开放技术详解/6.3-EPLB与LPLB专家副本和动态均衡.md)、[6.4](../guides/模块七-DeepSeek开放技术详解/6.4-DeepSelect稀疏索引与TopK算法.md)、[6.5](../guides/模块七-DeepSeek开放技术详解/6.5-DeepJIT编译缓存与运行时契约.md)、[6.6](../guides/模块七-DeepSeek开放技术详解/6.6-Ascend后端与跨硬件能力矩阵.md) |
| 7 | 3FS pending/committed 与文件长度、smallpond 分区恢复、FireFlyer/HAI 归属、DSec 公开报告与未公开实现的界线。 | [7.1](../guides/模块七-DeepSeek开放技术详解/7.1-3FS存储架构一致性与训练IO.md)、[7.2](../guides/模块七-DeepSeek开放技术详解/7.2-smallpond数据处理分区与恢复.md)、[7.3](../guides/模块七-DeepSeek开放技术详解/7.3-FireFlyer集群与通信存储协同.md)、[7.4](../guides/模块七-DeepSeek开放技术详解/7.4-DSec沙箱与Agent训练环境Infra.md) |
| 8 | VL/Janus/OCR 视觉数据流、Harness Agent 生命周期、Recipe 消息协议；不把 mock server 当完整托管服务。 | [8.1](../guides/模块七-DeepSeek开放技术详解/8.1-DeepSeekVL与Janus多模态路线.md)、[8.2](../guides/模块七-DeepSeek开放技术详解/8.2-DeepSeekOCR与OCR2视觉压缩.md)、[8.3](../guides/模块七-DeepSeek开放技术详解/8.3-DeepSeekHarness插件与Agent执行链.md)、[8.4](../guides/模块七-DeepSeek开放技术详解/8.4-DeepSeekRecipe协议格式与可复现接入.md) |
| 9 | 14 个标准库测试、MLA CPU/CUDA 数值与梯度对照、官方固定 EPLB CPU 规划；明确未执行大模型和集群复现。 | [9.1](../guides/模块七-DeepSeek开放技术详解/9.1-DeepSeek数学状态与MLA本地实验.md)、[9.2](../guides/模块七-DeepSeek开放技术详解/9.2-DeepSeek复现矩阵与工程验收.md) |

当前共 48 篇文档（9 篇导读、39 篇详解）。导读用于连接模块，不能替代详解的推导、源码和练习。

## 2. 来源与版本冻结

- [系统仓库快照](./deepseek-source-snapshots.json)：24 个固定仓库版本，包含代码许可与源码入口。
- [架构与模型快照](./deepseek-architecture-sources.json)：模型配置/权重 revision、仓库与论文，以及实际读取范围。两份清单有重复项目，不将条目数直接相加当成独立仓库数。
- 每篇正文列实际依据与适用条件。历史版本用于解释演变，当前仓库接口与 DeepInfer 的 vendor lock 必须分别核对。
- `HFAiLab/hai-platform`、TileLang 等相关项目保留原归属；不将合作、采用或底层依赖一律算作 DeepSeek 开源。

## 3. 跨作者复核后修正的关键边界

| 主题 | 最终采用的准确边界 |
|---|---|
| V4.1 / 1017 | 分离模型结构、权重更新、最小参考实现与生产执行路径；不由代码相同推断权重相同。 |
| 参数量 | 主干与 Engram 的数量分开，视觉和 DSpark 需另定计数口径。 |
| CED / Prefix replay | 有界 SWA replay 是近似；公开最小模型前向不等于已实现生产的完整跳层/分层索引节省。 |
| mHC | Single-Pass 不表示删掉 Sinkhorn；混合时序、读写次数与稳定性逐项说明。 |
| DSpark | 区分同步教学决策和使用历史置信度决定容量的生产调度；验证随机性不能反向改变本轮草稿决策。 |
| FlashMLA | 当前支持的硬件与 legacy 路径分开；索引、FP8/FP4 字节布局、空行 sentinel、CUDA/Ascend 算子能力分开。 |
| DeepEP | V2.5 与 V1/NVSHMEM 的实现不同；capacity 不是有效 token 数；deferred wait 与缓冲区寿命是调用契约。 |
| DeepInfer | 公开离线推理核心不等于完整在线服务；延迟输出必须 flush，复用输出需复制，并按 vendor lock 核对依赖。 |
| DeepGEMM | 最小 IO 字节是理想流量估计，对应算术强度是乐观估计，不能称为 AI 下界。 |
| DeepJIT | 已跟踪的头文件摘要可按 runtime 缓存、post-hook 摘要可按线程缓存；源码编辑后须重建 runtime/进程。 |
| 3FS | 强一致的 chunk 数据路径与活跃文件长度更新口径不同；pending/retry、relaxed read、fsync/close 分开讨论。 |
| smallpond | 固定实现使用 `by_rows`；任务 checkpoint 不自动保证用户外部副作用 exactly-once。 |
| ESFT | top-p 为累计路由质量而非固定专家百分比；冻结参数不等于把整个模块放进 no_grad。 |
| Recipe / DSec | Recipe 是协议编码/模板工具；DSec 在本次核对范围有公开报告，未发现对应完整官方运行服务源码。 |

## 4. 实验和未运行边界

- 可复查代码与结果在 [labs/deepseek](../../labs/deepseek/README.md)，包括标准库 oracle、MLA float64 前向/反向 CPU 与 RTX 3090 小张量实验、固定 EPLB CPU 规划。
- MLA CUDA 实验是普通 PyTorch 数值等价实验，没有运行 FlashMLA/DeepGEMM 的 SM100 专用算子。
- 未下载全部模型权重、未重训 DeepSeek、未部署 3FS/HAI/DSec、未完成多节点 RDMA 或 Ascend 实测；正文保留必要的目标环境步骤和验收标准。
- 论文性能数字归属于其报告环境，不推广为当前 RTX 3090 或任意集群的结果。

## 5. 网站交付

九章通过 `deepseek-open` 分类进入 `/AIInfraGuide/deepseek/`。正文中的源相对 Markdown 链接由站点插件转成已发布路由；构建、页面/锚点/图片/公式、搜索和浏览器验收见 [VALIDATION.md](./VALIDATION.md)。本清单不把“文件已经写出”视为浏览器验收通过。
