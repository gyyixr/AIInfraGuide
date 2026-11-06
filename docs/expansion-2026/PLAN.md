# AI Infra深度扩展：工作包与交付记录

更新日期：2026-11-07。

目标保持为：消化用户指定的三份参考材料，按主要章节、机制与源码不变量大幅扩充现有学习网站，并新增经一手资料核对的DeepSeek开放技术模块。深度由推导、完整算例、源码与版本边界、实践及带答案的问题判断，不由文件数量代替。

## 1. 实施前基线

- 工作分支main，基线`3ad88e4`，已合入上游`c09d316`。
- 当时为5个学习模块；分布式11章/23专题，RL10章/20专题。
- 推理第6—9章较提纲化；性能工程尚无独立正式模块。
- 本节描述实施前状态，不作为当前模块数量。

## 2. 三份参考材料及版本

| ID | 材料 | 原资料截点 | 证据约定 |
|---|---|---|---|
| training-infra | LLM超大规模与超长上下文训练Infra技术综述_2026 | 2026-08-24 | 全链路主题与设计约束，动态能力另核对原始来源 |
| megatron-source | Megatron-LM-20260829/ANALYSIS_REPORT | 2026-08-29 | 冻结`48a887f…`源码；正式0.19文档/tag单列 |
| inference-engineering | LLM推理工程师面试宝典_2026 | 2026-08-30 | SGLang `46d9427b…`、vLLM `9c8e90eb…`及235题身份 |

完整文件位置、SHA-256、行数与标题映射见[reference-manifest.json](reference-manifest.json)。技术事实使用对应论文、官方文档和固定源码；旧材料日期不自动代表本轮软件版本。

## 3. 已落盘工作包

| 工作包 | 当前内容与状态 | 入口 |
|---|---|---|
| T1 数据/源码/可靠性 | 分布式12—14章正文完成：治理、三索引、迭代、ModuleSpec、grad/optimizer、checkpoint与故障 | [训练数据](../guides/模块三-分布式训练/第12章-训练数据系统.md)、[Megatron状态](../guides/模块三-分布式训练/第13章-Megatron源码与训练状态.md) |
| T2 长上下文/设计 | 9.3—9.5、15—16章完成：位置、packing、有效评测、五类配置、课程/scaling、题与项目 | [成本设计](../guides/模块三-分布式训练/第15章-超大规模训练设计与成本.md)、[项目](../guides/模块三-分布式训练/第16章-训练源码题库与项目实战.md) |
| T3 核心缺口补齐 | 5.3、8.3、9.6、13.5/13.6、17章及13.1/13.4补充完成 | [框架与异构生态](../guides/模块三-分布式训练/第17章-训练框架与异构加速生态.md) |
| I1 推理工程 | 6—9章详解完成：容量、WideEP、PD、KV协议、Benchmark、背压、路由与故障 | [分布式推理](../guides/模块四-推理优化/第6章-分布式推理.md) |
| I2 源码与面试 | 11—12章完成，含235题逐题要点、结构化输出/投机分支和分岗mock | [源码数据流](../guides/模块四-推理优化/第11章-SGLang与vLLM源码数据流.md)、[题库项目](../guides/模块四-推理优化/第12章-推理源码题库与项目实战.md) |
| P 性能工程 | 独立模块六4章/8专题完成：模型、统计、GPU、编译、网络、NUMA、正确性与项目 | [性能工程](../guides/模块六-性能工程与生产实践/第1章-性能模型与实验方法.md) |
| RL长轨迹补充 | 第11章完成：mask/CP-DP、GSPO几何ratio、异步环境与reward版本 | [长上下文与异步RL](../guides/模块五-LLM-RL后训练/第11章-长上下文与异步RL工程.md) |
| D1 模型与证据 | DeepSeek1—2章完成，含已核对的早期路线、架构演进、MLA/MoE/稀疏/mHC/CED/Engram | [开放技术版图](../guides/模块七-DeepSeek开放技术详解/第1章-开放技术版图与版本证据.md) |
| D2 训练与后训练 | DeepSeek3—4章完成：FP8、DualPipe、Muon、MTP/DSpark目标、GRPO/R1/数学/Agent与蒸馏 | [训练系统](../guides/模块七-DeepSeek开放技术详解/第3章-预训练数值与并行系统.md) |
| D3 推理与算子 | DeepSeek5—6章完成：FlashMLA、DeepInfer、DSpark、GEMM/EP/均衡/TopK/JIT/Ascend | [推理运行时](../guides/模块七-DeepSeek开放技术详解/第5章-推理算子与DeepInfer运行时.md) |
| D4 存储与生态 | DeepSeek7—9章完成：3FS、smallpond、FireFlyer、DSec、视觉/OCR/Harness/Recipe与实验 | [存储与集群](../guides/模块七-DeepSeek开放技术详解/第7章-存储数据与集群基础设施.md)、[复现验收](../guides/模块七-DeepSeek开放技术详解/第9章-本地实验与复现验收.md) |
| S 文档集成 | README、既有路线、两个新learning-path页面、更新日志与本计划收尾 | [进阶导航](../guides/训练推理与DeepSeek进阶学习导航.md)、[来源覆盖网页](../guides/资料来源版本与覆盖边界.md) |

## 4. 覆盖验收如何解释

[训练覆盖审计](TRAINING_COVERAGE_AUDIT.md)将两份训练参考的主要章节/模块映射到正文，并保留深入、部分、提及与未覆盖的差别。

[推理覆盖审计](INFERENCE_COVERAGE_AUDIT.md)与[235题台账](inference-235-coverage.json)核对原ID、题干、分类、顺序、逐题要点及正文入口。

本轮资料扩展验收以三份参考的主要章节承接、核心机制深度、版本证据和可浏览交付为目标。不能将其表述为每个研究项目均有独立教程、原Megatron报告所有126问均已逐题实现，或全部项目已完成GPU/多机部署。

审计中的进一步研究和真实部署边界继续保留。它们不是要求本次必须安装所有厂商SDK、下载所有大模型或复现所有论文集群。

## 5. 已有实验与实际边界

| 实验层 | 现有产物 | 能支持的结论 |
|---|---|---|
| 标准库数学/状态 | 分布式、RL、DeepSeek labs及正文oracle | 索引、代数、概率、队列与状态不变量 |
| CPU/Gloo | 分布式两进程参考结果 | 给定数据的collective、累积、裁剪与AdamW更新 |
| 官方CPU规划 | 固定EPLB源码及结果 | 指定逻辑负载下的副本/放置计划 |
| 普通PyTorch MLA | CPU与RTX3090/SM86 float64参考结果 | 合成张量的完整/吸收路径前向与梯度一致 |
| 专用kernel/完整模型/生产集群 | 各专题给出验收矩阵与运行条件 | 不作为已在本机完成的结果 |

实验入口：`labs/distributed/README.md`、`labs/llm_rl/README.md`、`labs/deepseek/README.md`。其中的历史VALIDATION页面按其注明范围解释，不自动覆盖全部新增页面。

新增训练核心补齐的18个CPU oracle已通过；推理/性能和DeepSeek有各自独立检查记录。作者自检与主任务的集中站点验收分开记录。

## 6. 网站交付检查

当前为7个学习模块，加学习路线、面试与博客入口。

- 已接入模块导航、章节目录、上下篇和搜索生成流程。
- 文章间使用源相对`.md`链接，由现有站点插件转换为模块路由。
- `docs/guides`教程与`docs/posts`更新日志可在本地网站阅读。
- `labs`源码、原始审计JSON、模型与工具执行需要终端或相应环境；不承诺任意仓库文件都有网页路由。
- 本地URL保留`/AIInfraGuide/`，启动命令为`npm run site:local`。

最终构建与检查已经通过：730个网页全部HTTP 200且与当前构建逐字节一致；164281次本地链接和3808次本地资源引用检查无错误；28个代表页面、33个响应式场景及题库/搜索点击通过。历史断链已修复。完整范围、实验边界与报告入口见[VALIDATION.md](VALIDATION.md)。

影响交付的断链、公式/图渲染、搜索与页面问题应修复后再结束。硬件未验证和进一步研究项以明确边界交付，不用假运行填补。

## 7. 持久证据入口

- [原材料清单](reference-manifest.json)。
- [训练覆盖审计](TRAINING_COVERAGE_AUDIT.md)、[推理覆盖审计](INFERENCE_COVERAGE_AUDIT.md)。
- [训练框架与论文快照](training-framework-source-snapshots.json)。
- [DeepSeek仓库快照](deepseek-source-snapshots.json)、[模型与论文来源](deepseek-architecture-sources.json)。
- [文章更新日志](../posts/update-log.md)。

`.local/`和`/tmp/`中的抓取缓存、构建日志和截图用于本次工作复核，不替代正文中可追溯的一手链接，也不作为长期唯一实验依据。
