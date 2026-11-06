# 训练参考资料到当前网站的覆盖审计

审计日期：2026-11-07。范围为用户指定的两份训练参考，以及当前 `docs/guides/` 中可交叉映射的训练、RL、性能工程、CUDA和DeepSeek正文。审计只读正文，新增本文件；不把审计过程变成未经研究的新教程。

## 1. 审计结论与口径

**核心训练主线已有成体系的正文覆盖，但两份参考尚未达到所有主要分支同等深入的完整迁移。** 数据治理与三索引、TP/PP/CP/EP基础、main_grad与DistOpt、checkpoint/RNG、可靠性、长上下文、低精度、五类配置、RL和项目路线都有具体机制、公式或状态图及算例。不能据此把框架生态、其他加速器或Megatron全部版本门槛也标为完成。

本次审计随后驱动一轮限定补齐：HSDP/HFSDP、多实例DistOpt、AutoSP/HexiSeq、MCore MLA/MTP/PGCollection/动态调度、0.19两类known issues，以及TorchTitan/MaxText/六个开放框架/AMD-TPU-Trainium均已新增或扩写并复核。剩余缺口主要集中在部分TE/MoE高级分支、具名可靠性与数据前沿研究、非DeepSeek训练案例、原126问题库完整映射及真实项目执行，详见第5节。

评分采用以下定义，**不是按文件名、篇幅或出现次数计算覆盖率**：

| 标记 | 判断标准 |
|---|---|
| D：深入 | 抽读发现机制或数据流，至少一项推导/完整算例/明确状态协议，以及适用边界；治理类主题允许以schema与可追溯例子代替数值公式 |
| P：部分 | 有实质解释，但来源中的关键分支、实现条件、比较维度或验收尚缺 |
| R：提及 | 只在表格、题目、参考链接或短提示出现，不计为深入覆盖 |
| N：未找到 | 全站关键词检索结合相关正文抽读，没有找到对应机制内容；不把相邻主题冒充覆盖 |
| V：运行证据另计 | 正文设计与真实执行是两回事；CPU模拟、单GPU数学验证、官方kernel、多机生产证据分别列出 |

本审计未重新完成网站构建、逐页浏览或全部代码运行。目录中的历史验证文件只证明其注明的当时范围，不能外推为当前全部新增页面已验收。来源中的性能数字、最新能力与roadmap也不因写进矩阵而成为本机验证。

### 1.1 来源身份

- **S1：训练Infra综述**：`/home/gyy/python/LLM算法学习/LLM超大规模与超长上下文训练Infra技术综述_2026.md`；2352行；SHA-256 `c88a71265707a6ca32bfca10e83600803120834c9b8c236dd6cecb930c76c6f6`。
- **S2：Megatron报告**：`/home/gyy/python/LLM算法学习/Megatron-LM-20260829/ANALYSIS_REPORT.md`；4319行；SHA-256 `8d460d0a5256cd15ea5d5c5a365ca1549b4167ef8ee931dad6f8086b1635e467`。

S1材料截点为2026-08-24；S2材料冻结于2026-08-29。S2的源码基线是`48a887fecd01674346f724dcf44f417f455989f0`，0.19.0版本化文档/正式tag属于另一证据层。当前网站明确区分了二者，但未覆盖该报告的全部细节。

### 1.2 文章代号

`T`为模块三分布式训练，`R`为模块五RL，`P`为模块六性能工程，`D`为模块七DeepSeek。矩阵中的每个代号都是可点击文件链接，完整路径见文末索引。章节号指文章内的小节，不是源资料的章节号。

## 2. S1训练Infra综述逐主题矩阵

### 2.1 全景、成本与优化算法（来源§1—3）

| 来源段 | 网站正文 | 已抽读的实质证据 | 等级与剩余边界 |
|---|---|---|---|
| §1.1技术栈、§1.2迭代 | [T13.1]、[P1.1]、[T14.1] | provider→schedule→loss closure→grad finalize→optimizer→参数AG；异步完成边界与关键路径公式 | D；不是完整Megatron所有条件分支覆盖 |
| §1.3训练生命周期 | [T15.3]、[R1.1]、[R7.1] | 阶段合同明确数据、optimizer续用、LR、位置recipe与验收；RL角色闭环 | D；没有一个统一贯通全部阶段的已执行LLM训练run |
| §2.1 Dense FLOPs | [T15.1]§3—11 | 从GEMM三个阶段推6PN，GQA/SwiGLU参数手算，1M序列attention额外项 | D；注明causal/full-square、LM head、重计算口径 |
| §2.2 Attention内存 | [T9.1]§2、[C-FA1]、[C-FA2] | score与QKV分别计字节，Flash避免物化但不消除二次计算 | D；FA3/4代际实现另见缺口G5 |
| §2.3状态、§2.4激活 | [T13.3]§2、13；[T8.2]；[T11.2] |18与6+12/d字节、grad full buffer与有效range区分；时间上的存活峰值 | D；HSDP/HFSDP不在此覆盖之内 |
| §2.5 MFU/时间/goodput | [T15.1]§15—16、[T14.3]§10—12、[P1.1] |70B/3T/1024卡假设预算35.60天；6小时goodput与durable年龄手算 | D；数字均带假设，不是采购或硬件性能事实 |
| §2.6 MoE总/活跃参数 | [T15.1]§7、[T15.2]§11—13 |64专家top-2的总/活跃矩阵计数；checkpoint按总参数 | D |
| §3.1 Decoder block | [T13.2]§4—9、[D2.1]、[D2.2] |pre-norm残差、QKV、GQA、SwiGLU逐shape与参数例子；MLA独立展开 | D；MCore MLA特定后端gate另由[T13.5]补齐 |
| §3.2 AdamW/Adafactor/Muon | [T15.3]§3—6、14—16；[D3.3] |AdamW clip后[0.899,-2.098]；Adafactor行列统计小矩阵；Muon极分解方向 | D于AdamW/Muon主线；Shampoo仅状态/运算概述，LAMB/LARS未深入 |
| §3.3 LR/schedule | [T15.3]§8—10 |token warmup从32步变8步、cosine/linear公式、WSD分支与续训状态 | D；没有真实多模型schedule消融 |
| §3.4 scaling/μP | [T15.4]§2—18 |固定C=kPN推P/N最优幂律；归一化IsoFLOP例子；函数更新随宽度增长的反例 | D于推导与实验设计；完整Transformer μP参数分类缩放实现仍P |
| §3.5数值稳定 | [T15.3]§11—18、[T8.3]§17—20、[P4.1] |首异常tensor、QK/logit/scale探针；BF16 shadow与forward/grad/update分级验收 | D；QK clip/soft-capping各变体未做完整配方复现 |
| §3.6 MTP | [D3.4]、[D5.3]、[T13.5] |MTP移位label、辅助目标、DSpark区别；新增MCore eh_proj/roll/detach/PP与CP文档冲突 | D于模型数学与指定MCore合约；无官方多卡MTP运行 |

### 2.2 MoE与多维并行（来源§4—5）

| 来源段 | 网站正文 | 已抽读的实质证据 | 等级与剩余边界 |
|---|---|---|---|
| §4.1—4.2路由闭环 | [T10.1]、[D2.2]、[D6.2] |top-k→permute→dispatch→expert→combine；两token/三expert输出5与7.2，gate重复反例 | D |
| §4.3负载均衡 | [T10.2]§2—5、[D6.3] |aux loss与bias更新区分；真实EPLB CPU规划[4,2,1,1]及映射检查 | D；Quantile Balancing仅发布项提及，机制未迁移，见S2矩阵 |
| §4.4 capacity/dropless/GEMM | [T10.2]§6—8、[D6.1]§6—8 |counts [3,2,4,3]在capacity3/4下drop/pad表；M-axis/K-axis grouped区别 | D；Tutel仅无深入实现，不能算完整项目对比 |
| §4.5—4.6 EP/ETP/EDP与放置 | [T11.1]、[T15.2]§11、[D6.2]、[D6.3] |dense TP4×DP16与expert ETP1×EP8×EDP8共享64卡；物理副本和梯度归还生命周期 | D于双视图与拓扑；MCore完整group构造细节仍P |
| §4.7公开模型案例与Piper | [D1.1]、[D2.2]、[D3.3]、[T15.2] |DeepSeek家族深入；万亿MoE设计题有总/活跃参数与A2A账本 | P；Kimi/GLM/LongCat/Qwen训练配置比较表、Piper方法未系统迁移 |
| §5.1总体关系 | [T11.1]、[T13.1]§4、[T16.1]题1—3 |8卡rank坐标与组；SP不额外乘world，EP双视图 | D |
| §5.2 DDP与累积 | [T3.2]、[T4.3]、[R11.1] |有效元素3/5的Gloo参考记录；通用reducer alpha推导，DP2CP2梯度3/错误0.75与6 | D；GPU/NCCL未由CPU结果证明 |
| §5.3 ZeRO/FSDP1/FSDP2/HSDP | [T5.1]、[T5.2]、[T5.3]、[T13.3] |DTensor与reshard；新增HSDP/HFSDP/partial DistOpt状态表、d=D/I、2.25/3/4.5/7.5/9GB账本与两级归约oracle | D于来源状态主线；FSDP1内部实现与各后端性能仍未全部比较 |
| §5.4—5.5 TP/SP | [T6.1]、[T6.2]、[T13.2] |column dX SUM、row bias只加一次、SP AG/RS布局与shape | D |
| §5.6 PP/bubble | [T7.1]、[T7.2]、[T13.1]、[D3.2] |GPipe/1F1B/VPP、bubble两个分母、DualPipe通信重叠与不等时stage | D于数学与主调度；[T13.1]§12.2—12.4已补MCore P2P/pseudo-deallocation/combined |
| §5.7—5.10 CP/Ring/Ulysses/USP | [T9.1]§5—10、[T9.2]§1—9 |四rank KV轮转、online softmax完整例、GQA head约束、causal zigzag与通信口径 | D于Ring/Ulysses；[T13.6]补动态hybrid CP，[T9.6]补非均匀CP/HP；USP完整实现仍P |
| §5.11通信字节与时间 | [T2.2]、[P3.1]§2—10、[T13.3]§14 |8-rank 1GB/4KiB的alpha-beta数例；RDMA registration→completion→retire | D；各NCCL算法源码未全部覆盖 |
| §5.12并行选型 | [T15.2]、[T16.2]项目8 |合法候选过滤、状态下界、真实group、恢复gate与成本校准 | D于方法；未交付通用自动最优搜索器 |

### 2.3 长上下文、低精度与算子（来源§6—8）

| 来源段 | 网站正文 | 已抽读的实质证据 | 等级与剩余边界 |
|---|---|---|---|
| §6.1四种长度 | [T9.3]§1、[T9.5]§1—2 |配置/系统/训练/有效长度分离，任务阈值与置信要求 | D |
| §6.2五个瓶颈 | [T9.1]、[T9.4]、[T15.2]§14—15 |计算/激活/CP/输入/质量分账；1Mi-token网格与成本转移 | D |
| §6.3—6.4 RoPE与扩展 | [T9.3]§2—8 |二维旋转推导、0→0.7071插值例、base与统一缩放差别 | D于RoPE/PI；YaRN有分频直觉，PoSE/LongRoPE仅方法表，CLEX/LongRoPE2未展开 |
| §6.5 FlashAttention 1—4 | [C-FA1]、[C-FA2]、[D5.1]、[P2.1] |FA1/2的online softmax与work partition；FlashMLA缓存格式和接口 | P；FlashMLA不等于FA3/4，Hopper/Blackwell版FA专门机制不足 |
| §6.6 sparse attention | [T9.5]§8—12、[D2.3]、[D2.9] |边集合softmax、65536×1536稀疏边数、NSA/DSA与CSA2层级索引 | D于主要机制；MoBA只类型/引用层 |
| §6.7线性/SSM混合 | [T9.5]§13—15、[T13.4]§2—3 |标量recurrence/scan组合、状态跨段例1→2.5→4.25、Hybrid词表 | P；真实Mamba/GDN/线性attention训练scan/backward与分布式state并行缺 |
| §6.8长依赖数据 | [T9.3]§9—11、[T9.5]§3—5 |位置覆盖与依赖距离分离，多证据/代码/冲突任务，长token10%占attention64% | D于课程设计；真实长数据构建与消融未执行 |
| §6.9 packing | [T12.2]§7—9、[T9.4]§2—7、[R11.1] |S+1与cu_seqlens扣额外token；256/192/120 pairs；target作者mask | D |
| §6.10动态SP/ByteScale/AutoSP/HexiSeq | [T9.4]§9—15、[T9.6]§3—20 |Torch-IR三集合改写、SAC min-cut；ragged A2A、head子环、三阶段异构计划与容量过滤；实机/模拟边界拆开 | D于论文机制；没有作者系统运行、ByteScale完整实现仍P |
| §6.11—6.12评测与决策 | [T9.5]§2—7、15—18 |RULER/HELMET/NoLiMa矩阵；oracle与位置扰动；100题80%的Wilson区间 | D于评测方法；没有真实benchmark分数或数据集实现 |
| §7.1—7.2精度层次 | [T8.1]、[T15.3]§11、[P4.1] |参数/累积/master/moments分层与BF16小更新；敏感reduction边界 | D；TF32仅定位，未作硬件数值实验 |
| §7.3 FP8/amax | [T8.3]§3—8、16、[D3.1] |有限E4M3码本oracle、history scale将10还原为1；小值变零与block32保留 | D；oracle不是TE kernel |
| §7.4 MXFP8/NVFP4 | [T8.3]§8—15、[D6.1]、[D6.6] |8.25/4.5bit元数据账本、两级scale、随机舍入、转置block差异与二维scale | D于格式和recipe边界；新FP4论文的完整证明/收敛复现未覆盖 |
| §7.5低精度上线 | [T8.3]§17—20、[P4.1] |L0格式→L4长程质量；BF16 shadow、grad cosine/幅值与恢复 | D于验收设计；无完整FP8/FP4收敛运行证据 |
| §8.1 Roofline | [P1.1]§2—6 |理想96MiB/1365.33FLOP-byte例、HBM/L2/requested bytes分离、关键路径 | D |
| §8.2重点kernel | [P2.1]、[P4.1]、[D6.1]、[T6.2] |coalescing/bank/occupancy、RMSNorm backward有限差分、grouped GEMM与vocab CE | D于常见路径；每个kernel完整手写/架构实测并未全部提供 |
| §8.3编程编译栈 | [C-Triton]、[P2.2]、[D6.5]、[T9.6]、[T17.2]、[T17.4] |guard/Graph/JIT、AutoSP编译改写、JAX NamedSharding/XLA/Shardy与Pallas/NKI存储角色均有正文 | D于栈职责与主要机制；CUTLASS/CuTe/Pallas/NKI的完整kernel开发教程仍P |
| §8.4—8.6 collective/overlap/拓扑 | [T2.1]、[P3.1]、[P3.2]、[D6.2] |SUM/AVG、bytes口径、stream/event、NUMA、RDMA完成与资源回收、A2A矩阵 | D；[T17.4]已补RCCL与非NVIDIA通信合约，真实互操作仍未执行 |
| §8.7诊断树 | [P2.1]、[P1.2]、[T14.2] |先时间线再kernel/counter；同角色peer、误报与最早分歧 | D |
| §8.8加速器生态 | [T17.4]、[D6.6] |HIP warp/layout、hipBLASLt descriptor、RCCL；TPU ICI/DCN与Shardy；Neuron2.33 native/legacy与NKI；跨runtime桥接/恢复oracle | D于平台合约与迁移边界；未执行跨厂商训练或完整硬件性能验证 |

### 2.4 数据、可靠性与框架（来源§9—11）

| 来源段 | 网站正文 | 已抽读的实质证据 | 等级与剩余边界 |
|---|---|---|---|
| §9.1—9.2管线与治理 | [T12.1]§1—3、[T12.2]§12 |raw/normalized hash、source snapshot、parser/tokenizer版本schema，删除传播与权重处理边界 | D于可追溯设计；真实PII/许可处理器及治理流水未实现 |
| §9.3—9.4过滤去重 | [T12.1]§3—6 |过滤条件分布、Jaccard/MinHash方差、LSH候选概率与连通簇误删 | D；suffix-array/语义去重仅对比层，未有实现消融 |
| §9.5数据混合 | [T12.1]§7—8 |90/9/1B幂次采样，数学源22.244 epochs；样本率到有效token率公式 | D |
| §9.6 tokenizer | [F3.9]、[T12.1]§9、[T12.2] |embedding与token基础、hash/ID语义、扩词表与optimizer shape迁移 | P；Unicode/byte fallback、跨语言BPT/CPT分布、FIM/tool模板及迁移实验不足 |
| §9.7确定性数据恢复 | [T12.2]§3—13、[R11.1] |三索引完整token流、顺序与random sampler DP变更边界、已读/已提交区别 | D于GPTDataset；BatchWeave对象存储batch发布/GC协议未专门展开 |
| §9.8吞吐 | [T12.3]、[P3.2]、[D7.1]、[D7.2] |40MB/s却需23CPU核；163840 opens；NUMA/H2D状态机、3FS复制与shuffle4D | D于一般原理；OSDI生产三类跨数据中心I/O案例未独立复盘 |
| §10.1故障分类 | [T14.2]§2—6 |crash/hang/fail-slow/SDC/数据/控制面区分，p^N与100/180ms例 | D |
| §10.2—10.4 checkpoint/重分片 | [T14.1]§2—14、[T13.3] |state tree、staging→write→metadata→tracker，12元素3份→2份范围交集与optimizer gate | D；无真实分布式存储断电/跨TPPP恢复实验 |
| §10.5间隔 | [T14.1]§15、[T16.1]题38 |C/I+I/(2M)求导，8秒/7200秒得到339.41秒；异步与RPO边界 | D |
| §10.6自动恢复 | [T14.2]§8—17、[R11.3]§19 |rerun fatal/nonfatal、NVRx组件、inprocess清理、170秒恢复关键路径 | D于通用/NVRx；TrainMover/TorchFT运行态迁移与算法语义未深入 |
| §10.7 fail-slow | [T14.2]§4—6、[T14.3] |同角色peer中位数、计算/通信/输入隔离、平台计数关联 | D于诊断结构；GREYHOUND/Holmes/Minder分别的算法只引用或未展开 |
| §10.8 SDC | [T14.2]§7—12、[P4.1] |hash/finite/重放覆盖不同错误，named RNG与浅拷贝边界 | P；AEGIS在线检测、OpGuard语义指纹、workload replay细节未系统迁移 |
| §10.9—10.10观测/集群 | [T14.3]§2—9、13—19、[P3.1] |逻辑rank到UUID/NIC/NUMA；RPO360秒、goodput1.9M；Slurm术语和隔离演练 | D于方法；没有可部署Slurm/K8s训练调度完整配置 |
| §10.11系统案例 | [D7.3]、[D7.4]、[T15.2]、[T16.2] |FireFlyer与DSec有机制/数字/边界；系统设计与校准路线具体 | P；MegaScale/Di-PS/FLARE/SimAI/TrainMover的各自设计未同等展开 |
| §11.1训练框架版图 | [T17.1]、[T17.2]、[T17.3]、[T13.1]、[T5.2]、[R7.2] |TorchTitan/MaxText固定新入口；六开放框架逐项对象、并行、checkpoint、维护状态和验收，不只是名称表 | D于主要生态对象；NeMo/Megatron Bridge产品层的完整训练/转换流程仍P |
| §11.2—11.3层级与选型 | [T17.1]—[T17.4]、[T15.2]、[D1.2]、[R7.2] |框架/编排/plugin/kernel职责、相同batch字段的token单位、weights/full resume反例与逐平台能力矩阵 | D于任务驱动选型；没有六框架统一GPU benchmark |
| §11.4版本合同 | [T13.4]§1、[T16.2]§2—3、[D1.2]、[P4.1] |固定SHA、包版本/tag分层、依赖/容器/硬件/误差/验收记录 | D；当前全部页面交付检查由root另行验证 |
| §11.5 Mesh/DTensor/GTP | [T5.2]、[T5.3]、[T13.4]、[T17.1]、[T17.2]、[T17.4] |DTensor与uneven unit、GTP per-weight remat、TorchTitan声明式分片、JAX logical axes/NamedSharding与Shardy说明 | D于主抽象；不是所有redistribution/编译优化的完整实现审计 |

### 2.5 RL、设计、前沿与生产（来源§12—20）

| 来源段 | 网站正文 | 已抽读的实质证据 | 等级与剩余边界 |
|---|---|---|---|
| §12.1 SFT/DPO | [R1.1]、[R1.2]、[R4.1]、[R4.2] |动作mask、DPO相对logprob与偏好目标推导；与RL行为概率分开 | D |
| §12.2—12.3角色/PPO/GRPO | [R2.1]、[R2.2]、[R3.1]、[R3.2]、[R7.1] |PPO clip、GAE、组优势、DAPO/DrGRPO归约及角色数据流 | D；RLOO、reward scaling边界明确 |
| §12.4—12.5共置/异步 | [R8.1]、[R8.2]、[R11.2]、[R11.3] |固定GPU预算比较、q/old/current两段比率、GSPO几何均值不等于无偏IS；队列/取消/reward版本 | D；完整异步框架GPU运行未执行 |
| §12.6框架 | [R7.2]、[R10.2] |verl V1、ROLL role与slime generate/train/update_weights固定源码入口 | P；NeMo RL/OpenRLHF/DeepSpeed-Chat未按相同深度展开 |
| §12.7长上下文RL | [R6.1]、[R11.1]、[R11.2]、[R11.3] |工具观测mask、shift、CP/DP alpha归约、实际前缀、奖励上下文/超时与选择偏差 | D；6个独立CPU oracle可复核，不是多卡收敛结果 |
| §13.1—13.3合同/容量/候选 | [T15.1]、[T15.2] |参数矩阵、18字节状态、attention项、world与expert双视图、概念配置账本 | D |
| §13.4—13.5 benchmark/恢复验收 | [T16.2]、[P4.1]、[P4.2] |reference→单步→目标硬件→故障canary；输入输出、断言、反例和回退 | D于验收设计；多数项目是计划而非已完成实现 |
| §13.6配置搜索 | [T15.2]§16—18、[T16.2]项目8 |feasibility先于成本排名、unknown边界、exhaustive oracle与trace校准 | P；只有容量小脚本，未实现完整搜索器 |
| §14.1 70B常规长度 | [T15.2]§5—6 |64卡A/B状态26.25/39.375GB、microbatch128/64与bubble差异 | D，教学配置 |
| §14.2 70B128K | [T15.2]§7—8 |256卡网格、CP不自动切参数、d4/d16两种optimizer域与长度桶 | D，未实跑 |
| §14.3 400B Dense | [T15.2]§9—10 |1024卡23.4375GB状态下界、7.2TB账本与stage不均 | D，未实跑 |
| §14.4 1T MoE | [T15.2]§11—13 |total/active、64卡dense/expert两网格、路由尾部与optimizer/ckpt | D，未实跑 |
| §14.5百万上下文 | [T15.2]§14—15、[T9.5] |1Mi-token/CP32局部32768、技术各自解决和未解决项 | D，未实跑 |
| §14.6决策记录 | [T15.2]§18、[P1.2]、[P4.2] |候选淘汰、预测残差、trace、数值/恢复与失败方案 | D |
| §15.1—15.3能力矩阵 | [T16.1]、[T16.2]、[P4.2] |40完整问答及跨数学/源码/状态/性能/恢复八问 | P于来源全量高级能力；没有把所有P2研究路线发展为教程 |
| §15.4十二周 | [T16.2]§12 |逐周项目与必须交付；资源不足时分级完成，不冒称GPU证据 | D |
| §15.5源码顺序 | [T13.1]、[T13.2]、[T13.3]、[T16.1]§4 |producer/consumer、shape、group、梯度、恢复有阅读入口 | D于主线；报告列出的全部函数覆盖不成立 |
| §16.1公开模型信号 | [D1.1]、[D2.6]、[D4.2]、[D4.4] |DeepSeek固定报告和开放代码版本细化 | P；非DeepSeek模型时间线/训练披露基本未迁移 |
| §16.2八方向 | [T9.4]、[T9.5]、[T8.3]、[T13.4]、[T14.2]、[R11.3] |MoE/长上下文/稀疏/低精度/动态mesh/可靠性/RL/数据在线化均有一般机制 | P；若干具名方法仍R/N，不能说八方向所有论文都已详解 |
| §16.3未解决问题 | [T15.4]§7—12、[T9.5]§15、[T8.3]§17—22 |IsoFLOP条件、有效长度、低精度长期质量、误差与跨硬件证据边界 | D于判断方法，未做新的开放问题研究 |
| §17.1开跑前 | [T15.2]§16、[T16.2]、[P4.1] |shape/数值/容量/恢复/拓扑/版本门禁及故障输入 | D |
| §17.2运行中 | [T14.3]§6—9、[T12.1]、[R11.3]§21 |分层指标、source/长度/队列年龄、checkpoint durable与最慢rank | D于设计，未部署持续dashboard |
| §17.3结束后 | [T14.1]、[T15.2]§18、[T16.2]§14 |完整状态与模型导出区分、决策/失败/恢复记录 | P；独立故障域最终复制、产物归档/删除执行与训练卡片未形成完整流程 |
| §18故障速查/OOM顺序 | [T12.3]、[T14.2]、[T15.3]、[P2.1]、[P4.1] |按阶段定位数据/通信/kernel/optimizer，显存对象与生命周期，不盲目调flag | D |
| §19术语表 | 各专题正文与[T16.1] |术语在具体数据流中解释，多处区分逻辑/物理、token/序列、参数/状态 | P；没有完整统一术语表及全部缩写交叉入口 |
| §20一手资料 | 各专题文末、[D1.2] |固定SHA/版本化文档/论文/实验边界有来源 | P；不要求逐字复制书目，但遗漏主题对应的一手资料未被正文消化 |

## 3. S2 Megatron报告的模块与附录矩阵

本节由独立只读子审计与主审合并。来源范围包括A/B/C/D主模块、公式底稿、正式问答、29项Coding矩阵、15项系统设计题、P1—P8项目和交叉验证。DeepSeek文章能支撑数学与公开模型原理，但不能替代特定MCore代码gate。

| 来源模块/段 | 网站正文 | 实际证据 | 等级与未覆盖项 |
|---|---|---|---|
| 0六层证据、1岗位能力、2生态、3不变量、4迭代全景（18—175行） | [T13.1]§2—5、[T13.4]§1、[T16.1]§1、[D1.2] |固定SHA、包字符串/tag/latest分离；owner/consumer与完成边界 | D于版本和不变量；岗位/生态全量比较仍P |
| A1—2编排与真实调用链（185—279） | [T13.1]§2—12.4 |主链与8卡例；新增dynamic GBS阈值转换、RL packing例外、P2P/pseudo-deallocation/combined调用边界 | D于主链与指定补齐分支 |
| A3拓扑（280—385） | [T11.1]、[T13.6]§2—10 |required set/legacy adapter、双RankGenerator、32卡网格、MultiModule与不同group消费者；独立枚举1296组合对固定函数 | D于主要PG合约；SHARP等平台优化仍需具体配置复核 |
| A4 TP/SP（386—471） | [T6.1]、[T6.2]、[T13.2]§6—9 |column/row forward/backward、bias、GQA和SP布局有数例 | D数学；不同文章固定commit不可互相代替源码证据 |
| A5 CP（472—527） | [T9.1]、[T9.2]、[T13.6]§11—17 |新增hybrid CP动态2幂分组、barrier、末subsample、placement和PP/graph/FSDP禁配 | D于冻结分支；未实跑动态PG或目标kernel |
| A6 microbatch（528—568） | [T13.1]§12.1、[T15.3]§8 |threshold按是否传seq_length解释、0起点/转换后严格递增、当前台阶整除、旧rampup弃用、RL packing例外 | D；已比源概括更精确地限定不下降断言 |
| A7 schedules/P2P（569—706） | [T7.1]、[T7.2]、[T13.1]§12.2—12.4、[T13.6] |奇偶提交顺序、group-local→global peer、shape交换、pseudo-deallocation和combined路径已补 | D于主要机制；全部schedule代码覆盖仍未宣称 |
| A8—15排障/34问/计算/设计/覆盖表（707—1302） | [T16.1]题1—14、[T16.2]项目1—3、[T15.2] |完整代表答案、参考对照、反例、pipeline项目验收 | P；不等于原34题逐题答案和原函数覆盖台账迁移 |
| B1—4 GPT/ModuleSpec/Config/Dense（1329—1525） | [T13.2]§2—9、[T13.1]§5—7 |函数不实例化、重复kwargs报错oracle；GQA/SwiGLU shape/参数；残差位置 | D主线；parallel_output推理shape、viewless、CrossAttention GQA拒绝、tied runtime-weight/TE限制仍P |
| B5 TE（1526—1566） | [T13.2]§10—12、[T8.3]、[P4.1] |tuple contract、expert通信职责、backend与hardware三道门、recipe与数值验收 | P；冻结TE逐项版本gate、UB buffer name、首microbatch weight cache等缺 |
| B6 MoE（1567—1683） | [T10.1]、[T10.2]、[D6.1]、[D6.2] |6token/4expert闭环、capacity、aux/z-loss、permutation、split与通信数例 | P于MCore细节；AllGather/AllToAll/Flex比较、ETP、topk1归一退化、TP>1强制SP等缺 |
| B7 MLA（1684—1733） | [D2.1]、[T13.5]§2—6 |吸收代数与GPU oracle；新增r_Q、cache_mla_latents dynamic-only、FlashMLA/block64、packed hybrid CP拒绝与TE gate | D于算法及指定MCore gate；官方kernel运行未验证 |
| B8 MTP（1734—1779） | [D3.4]、[T13.5]§7—17 |eh_proj TP gather/SP scatter、D+1 hidden、label/CP/THD roll、detach、VPP/physical placement及文档-tag冲突 | D于指定主线；没有真实CP/MTP训练或FP4收敛复现 |
| B9—14排障/40问/计算/设计（1780—2203） | [T13.2]、[T16.1]题7—8/15—18、[T16.2] |有完整代表题；DeepSeek模型专题补数学 | P；MLA/MTP/TE专属问答尚不能逐题映射 |
| C0—2梯度/optimizer（2208—2479） | [T13.3]、[T5.3] |主状态链；新增d=D/I、完整dtype理论表、intra RS→inter AR、communication stream等待与inner AG | D核心与多实例；optimizer factory/ChainedOptimizer/完整特殊finalize表仍P |
| C3 FSDP/ZeRO/HSDP/HFSDP（2480—2509） | [T5.1]、[T5.2]、[T5.3] |inner/outer状态与边界、HFSDP optimizer全分片、unit uneven DTensor与量化局部性 | D于固定官方文档合约；不冒充逐行审计全部M-FSDP分配实现 |
| C4 checkpoint（2510—2615） | [T14.1]§4—16 |common/sharded/factory、staging/finalize、3→2范围、optimizer格式、RNG与rerun gate | D主要合约；local/direct/legacy各后端全部细节未逐一复现 |
| C5数据（2616—2688） | [T12.2]、[T12.3] |三索引/S+1/THD、cache风暴、blend实际比例、random sampler DP变化 | D核心；RandomSeedDataset样本级seed公式与worker无关性缺 |
| C6 rerun/NVRx（2689—2753） | [T14.2]§8—14、[R11.3] |fatal/nonfatal exit16/17、同GPU/换GPU矩阵、RNG集合/浅拷贝、组件职责与abort | D；未执行真实故障恢复，正文已注明 |
| C7—11框架/36问/计算/设计（2754—3177） | [T16.1]题19—34/38、[T16.2]项目5—6 |四点探针、多步optimizer/reshard验收与故障反例 | P；[T5.3]已补HSDP/partial instances；precision-aware与factory等题仍不完整 |
| D1发布项（3182—3202） | [T13.4]§1、[D1.2] |旧快照与0.19文档分开，Experimental依赖一同描述 | P；Quantile Balancing、streaming quantized load、MFSDP v2状态未系统迁移 |
| D2 Hybrid（3203—3251） | [T13.4]§2—7 |pattern词表≠translator矩阵、3block→6positions、在线optimizer/离线weights-only | D；需要固定目标实现另做迁移实验 |
| D3 GTP（3252—3311） | [T13.4]§8—14 |TP×remat、16/64MiB持久/聚合、AG与FP32 wgrad RS、窗口暴露 | D于版本化合约；非冻结48a887f源码运行 |
| D4—8 known issues/16问（3312—3447） | [T13.4]§17、[T13.5]、[T16.1] |新增TE CE与2.18/native、NCCL2.30.5/SPCX payload、内部IB/HybridEP对照；MTP CP冲突 | P整体；两类known issues已D，Quantile/streaming load等只发布项说明 |
| 公式底稿0—11（3448—3689） | [T15.1]、[T10.2]、[T7.1]、[T7.2]、[T14.1]、[T5.3]、[P3.1] |新增三dtype状态表与d=D/I；其余batch/pairs/ring/CP/PP/MoE/MFU/间隔均有推导 | D于主要公式；每个特殊配置的实际字节仍需按实现重新列账 |
| 高频陷阱（3690—3772） | [T16.1]、[T13.3]、[T14.1] |40题各有答案/推演/边界；多处纠正state与版本混用 | P；不能因40题涉及相同大类就推断全部陷阱已覆盖 |
| 29项coding/15项系统设计（3773—3845） | [T16.1]、[T16.2]、[T15.2]、[P4.2] |数值参考、layout、状态机、验收及五类候选有内容 | P；PGCollection已新增完整机制/数例；RNG tracker parity、overflow hysteresis等仍尚非完整实现/答案 |
| P1—P8项目/答辩（3846—3908） | [T16.2]§4—14 |8项目分别有输入输出、不变量、验收、失败定位及12周路线 | D于项目规格；V不等于8个项目已完成 |
| 跨模块洞察（3909—3957） | [T13.1]、[T13.3]、[T13.4]、[T14.1]、[P4.1] |ownership、layout→collective、overlap生命周期、三种恢复等级均有反例 | D主思想；feature矩阵仍非完整 |
| Mock/10周计划（3958—4083） | [T16.2]§12—15、[T16.1]§3 |重组为12周路线与答辩，不是简单标题 | P；四套Mock及原10周对应表未逐项迁移 |
| 外部一手证据/交叉验证/质量说明（4084—4319） | [T13.4]、[T14.1]、[T14.2]、[D1.2]及各文末 |固定来源、CPU/GPU/生产边界、包字符串≠tag | P；源报告全部审计覆盖表和冲突记录未原样迁移 |

### 3.1 题库数量与源码版本的特殊说明

S2有A34、B40、C36、D16，共126道正式问答，另有计算题、29项Coding和15项系统设计。网站[T16.1]的40题均有完整答案和边界，新[T13.5]/[T13.6]补充MLA/MTP/PG问题，但仍尚无126题逐项映射，不得声称“原题全部回答”。[T16.2]的8项目是具体可执行规格与验收设计，不是8个已完成代码项目。

旧[T11.1]、[T7.1]、[T7.2]等正文使用另一个已注明的2026-10-30固定提交`7d6895ac...`。它们可以支撑数学和设计解释，但不能作为48a887f的逐条实现证据。新[T13.2]对重复`**kwargs`会报错的说明比源报告“合并参数”的概括更精确，应保留这一核对结果。

## 4. 可检查运行证据与尚未验证范围

| 产物 | 本轮查看到的证据 | 能支持的结论 | 不支持的外推 |
|---|---|---|---|
| `labs/distributed/VALIDATION.md`及reference JSON |记录CPU数学检查、真实两个CPU/Gloo进程的多步AdamW与局部均值反例 |基础collective数值、梯度/optimizer状态参考一致 |NCCL/GPU性能、多机存储事务、当前新增全部页面已浏览 |
| `labs/llm_rl/VALIDATION.md`及reference-run |历史记录CPU bandit、有限差分与40+40恢复等价 |小离散任务目标与CPU恢复闭环 |真实LLM、verl/ROLL/slime多卡训练或GSPO收敛 |
| `labs/deepseek/results/mla-cuda-reference.json` |scope明确普通PyTorch float64/RTX3090，logit/output/grad误差约1e-16/1e-16/1e-18，漏RoPE负对照0.0361 |合成MLA吸收代数及单GPU自动微分对照 |FlashMLA官方kernel、V3/V4.1整模型性能与收敛 |
| `labs/deepseek/results/eplb-reference.json` |固定EPLB CPU规划，副本[4,2,1,1]，模型GPU loads全60 |官方Python放置算法在给定输入可运行 |专家真实执行、迁移、NCCL/RDMA性能 |
| 新T8.3/9.3—9.5/12—16文章oracle |标准库数值与边界示例；本轮扩写已分别运行，执行日志在[training-doc-checks.txt](./evidence/training-doc-checks.txt)、[training-second-round-checks.txt](./evidence/training-second-round-checks.txt) |格式、索引、范围、概率、成本与更新数学 |全部MCore目标实现、真实长上下文训练或FP4 kernel |
| 新R11.1—11.3 |[rl-long-context-oracle-checks.txt](./evidence/rl-long-context-oracle-checks.txt)中6个CPU oracle成功；GSPO偏差有限反例 |mask、归约、几何ratio、队列容量的教学正确性 |Qwen实验复现或真实Agent平台吞吐 |

执行日志已归档到本目录的 `evidence/`；网站文章保留可运行代码和边界，持久实验结果位于 `labs/`。本审计没有重新执行上述 GPU 或分布式实验。

## 5. 确认缺口与后续补齐边界

以下保留审计发现和补齐验收要求，同时标明当前状态。它们不是将用户完整目标缩小到已完成部分；未关闭的分支仍保留。

| ID | 初审缺口与当前状态 | 来源与初审证据 | 补齐验收要求 |
|---|---|---|---|
| G1 | **主要对象已补齐**：T17.1—17.4；NeMo/Bridge产品流程仍P |S1§8.8/§11.1：相关框架名称多数全站无机制正文；Ascend覆盖不替代AMD/TPU/Trainium |逐项目明确对象、入口、并行与checkpoint；设备编译/collective/精度/调试/迁移案例；原始版本来源 |
| G2 | **已补齐**：T5.3 |S1§5.3，S2 C3及公式9；全站缺HSDP/HFSDP与num_distributed_optimizer_instances |inner/outer state表、intra RS→inter AR、d=D/I、内存/通信手算、真实group与恢复约束 |
| G3 | **已补齐**：T13.5 |S2 B7/B8及问答；DeepSeek数学文章不含这些符号 |cache_mla_latents、block64/动态推理、TE gate；roll_tensor/CP zigzag/THD、eh_proj、detach、PP placement与文档-tag冲突 |
| G4 | **主要协议已补齐**：T13.6/T13.1 |S2 A3/A5/A6/A7 |ProcessGroupCollection、多grid与required set、双RankGenerator；dynamic GBS、P2P次序/shape、hybrid CP禁配、pseudo-deallocation |
| G5 | **AutoSP/HexiSeq已补齐**：T9.6；其余细分方法仍P/R |S1§6.4/6.5/6.7/6.10：PoSE等多为表格，AutoSP/HexiSeq仅链接，FA3/4和真实scan不足 |按论文解释自动SP/异构分配；逐方法辨认改变表示/计算/布局的对象；完整数例与实现/硬件边界 |
| G6 | **两类known issues已补齐**：T13.4；其余高级分支仍P |S2 B5/B6/D4：有通用adapter/recipe但缺精确来源矩阵 |TE版本、UB/cache、dispatcher/ETP/SP限制、topk1边界；CE/NCCL问题与限定回退方案；不可宣称旧问题今日普遍存在 |
| G7 | 更广可靠性研究 |S1§10.6—10.11/§16 |TrainMover/TorchFT运行态/算法区别，AEGIS/OpGuard诊断对象和误报/开销；保留论文环境边界 |
| G8 | 数据前沿和tokenizer工程 |S1§9.6—9.8 |BPT/CPT与Unicode/byte fallback/FIM实际例；BatchWeave发布/恢复/GC；真实过滤与污染检查产物 |
| G9 | 非DeepSeek公开训练案例 |S1§4.7/§16.1 |Kimi/Qwen/GLM/LongCat按参数、token、长度、硬件、optimizer与披露等级比较；不得以产品名反推未披露指标 |
| G10 | 完整题库映射与项目执行 |S2正式126问/Coding29/设计15/Mock；现40题与8项目规格 |逐题对照列明已答/合并/缺失；补专属源码题；实际项目执行与纸面项目规格分开 |
| G11 | 结束后生产交付与统一索引 |S1§17.3/19/20 |训练卡片、最终独立故障域状态复制/校验、归档删除协议、术语表与来源导航 |

### 5.1 本轮限定补齐的复核记录

| 新增或扩写 | 新正文证据 | 验证层次 |
|---|---|---|
| [T5.3] |473行，HSDP/HFSDP/partial DistOpt、dtype表、d=D/I、两级stream归约与checkpoint |2个CPU oracle；固定M-FSDP文档+buffer源码 |
| [T9.6] |487行，实际读取AutoSP/HexiSeq v1方法与附录，Torch-IR/SAC、ragged A2A/head子环、三阶段搜索 |2个CPU oracle；作者系统未运行，实机/模拟范围分别标注 |
| [T13.5] |MLA运行gate、MTP完整shape、roll/THD/detach/placement、CP文档与tag冲突 |固定48a887f源码与0.19文档；CPU位置/梯度数例 |
| [T13.6] |PGCollection required set/legacy/MultiModule、双generator、动态hybrid CP与禁配 |独立枚举器1296种组合对固定纯函数；无真实多卡 |
| [T13.1]新增§12.1—12.4 |dynamic GBS、P2P奇偶与peer、shape、pseudo-deallocation、combined边界 |固定源码；microbatch不下降增加RL packing例外 |
| [T13.4]新增§17 |TE CE修复版本与NCCL2.30.5/SPCX已知问题 |本轮读官方0.19 release；未实机复现，不泛称新版仍有问题 |
| [T17.1]、[T17.2] |各408行，当前TorchTitan Python recipe/声明式分片与MaxText NNX/Orbax v1真实入口 |固定官方SHA；共4个CPU oracle；未安装框架 |
| [T17.3] |496行，六项目逐项入口、并行/恢复/适用与维护状态，非名称表 |6个官方SHA+关键源码；2个CPU oracle；Foundry归档状态本轮API确认 |
| [T17.4] |AMD/TPU/Trainium新旧栈、warp/layout、collective互操作、桥接与状态迁移 |官方版本化/滚动文档分开；3个CPU oracle，无厂商SDK或混合硬件执行 |

本轮集成时，8篇新详解的17个Python代码块加13.1新增dynamic GBS例子共18个CPU程序再次执行成功；当前11篇相关文件79个公式通过KaTeX解析。记录在[training-final-gap-fill-oracles.json](./evidence/training-final-gap-fill-oracles.json)；框架与论文身份持久记录在[training-framework-source-snapshots.json](training-framework-source-snapshots.json)。

同时更新第5、9、13章导读，并新增第17章导读。上述补齐关闭所列核心缺口的正文层，不把G5其余方法、G6其余高级分支、G7—G11自动关闭，也不把CPU oracle升级成多机性能证据。

## 6. 审计方法与可复核检索

1. 核对两份源文件的行数与SHA-256；按完整正文识别主机制和子分支。
2. 从当前`docs/guides`查找候选，抽读公式、状态流、算例和验证边界，而非只匹配title。
3. 对疑似缺口全站检索具名机制与实现符号，再检查相邻文章是否用同义表述实质覆盖。
4. S2由独立子审计复核，重点分开模型数学与Megatron实现gate。
5. 记录已有labs证据的scope，避免用CPU或普通PyTorch GPU替代官方kernel/多机验证。
6. 链接检查只验证本地映射可解析；外部引用可用性与最新版本不是本轮覆盖审计的全部验收目标。

可复核的检索词包括：`HSDP`、`HFSDP`、`ProcessGroupCollection`、`cache_mla_latents`、`mtp_detach_heads`、`roll_tensor`、`hybrid_cp_schedule`、`num_distributed_optimizer_instances`、`Quantile Balancing`、`AutoSP`、`HexiSeq`、`Pallas`、`NamedSharding`、`TorchTitan`、`MaxText`、`Trainium`、`AEGIS`、`OpGuard`、`BatchWeave`。零命中只作为辅助证据，不能单独替代正文阅读。

本文件不提供一个整体“百分之多少已覆盖”的数字：没有对源问答和全部代码合约建立一对一完整台账时，单一百分比会掩盖深浅差异。

## 7. 本站文件索引

下列代号用于前述矩阵，链接指向实际Markdown源文件。

| 代号 | 实际文件 |
|---|---|
| C-FA1 | [6.1-FlashAttention V1详解.md](<../guides/模块二-CUDA编程与算子优化/6.1-FlashAttention V1详解.md>) |
| C-FA2 | [6.2-FlashAttention V2详解.md](<../guides/模块二-CUDA编程与算子优化/6.2-FlashAttention V2详解.md>) |
| C-Triton | [7.1-Triton编程框架.md](<../guides/模块二-CUDA编程与算子优化/7.1-Triton编程框架.md>) |
| D1.1 | [1.1-DeepSeek模型与开源项目演进.md](<../guides/模块七-DeepSeek开放技术详解/1.1-DeepSeek模型与开源项目演进.md>) |
| D1.2 | [1.2-开源代码开放权重与论文披露边界.md](<../guides/模块七-DeepSeek开放技术详解/1.2-开源代码开放权重与论文披露边界.md>) |
| D2.1 | [2.1-MLA低秩压缩解耦RoPE与矩阵吸收.md](<../guides/模块七-DeepSeek开放技术详解/2.1-MLA低秩压缩解耦RoPE与矩阵吸收.md>) |
| D2.2 | [2.2-DeepSeekMoE细粒度专家与共享专家.md](<../guides/模块七-DeepSeek开放技术详解/2.2-DeepSeekMoE细粒度专家与共享专家.md>) |
| D2.3 | [2.3-NSA与DSA稀疏注意力.md](<../guides/模块七-DeepSeek开放技术详解/2.3-NSA与DSA稀疏注意力.md>) |
| D2.6 | [2.6-V4点1架构更新与原生视觉.md](<../guides/模块七-DeepSeek开放技术详解/2.6-V4点1架构更新与原生视觉.md>) |
| D2.9 | [2.9-CSA2与分层稀疏索引.md](<../guides/模块七-DeepSeek开放技术详解/2.9-CSA2与分层稀疏索引.md>) |
| D3.1 | [3.1-FP8细粒度缩放与训练数值链路.md](<../guides/模块七-DeepSeek开放技术详解/3.1-FP8细粒度缩放与训练数值链路.md>) |
| D3.2 | [3.2-DualPipe与DualPipeV流水调度.md](<../guides/模块七-DeepSeek开放技术详解/3.2-DualPipe与DualPipeV流水调度.md>) |
| D3.3 | [3.3-Muon优化器与V4训练配方.md](<../guides/模块七-DeepSeek开放技术详解/3.3-Muon优化器与V4训练配方.md>) |
| D3.4 | [3.4-MTP与DSpark训练目标.md](<../guides/模块七-DeepSeek开放技术详解/3.4-MTP与DSpark训练目标.md>) |
| D4.2 | [4.2-V3点2与V4点1AgentRL和环境系统.md](<../guides/模块七-DeepSeek开放技术详解/4.2-V3点2与V4点1AgentRL和环境系统.md>) |
| D4.4 | [4.4-蒸馏合成数据与1017自反馈更新.md](<../guides/模块七-DeepSeek开放技术详解/4.4-蒸馏合成数据与1017自反馈更新.md>) |
| D5.1 | [5.1-FlashMLA稀疏算子与KV缓存格式.md](<../guides/模块七-DeepSeek开放技术详解/5.1-FlashMLA稀疏算子与KV缓存格式.md>) |
| D5.3 | [5.3-DSpark投机验证与置信度调度.md](<../guides/模块七-DeepSeek开放技术详解/5.3-DSpark投机验证与置信度调度.md>) |
| D6.1 | [6.1-DeepGEMM低精度矩阵与MegaMoE.md](<../guides/模块七-DeepSeek开放技术详解/6.1-DeepGEMM低精度矩阵与MegaMoE.md>) |
| D6.2 | [6.2-DeepEP通信布局与异步生命周期.md](<../guides/模块七-DeepSeek开放技术详解/6.2-DeepEP通信布局与异步生命周期.md>) |
| D6.3 | [6.3-EPLB与LPLB专家副本和动态均衡.md](<../guides/模块七-DeepSeek开放技术详解/6.3-EPLB与LPLB专家副本和动态均衡.md>) |
| D6.5 | [6.5-DeepJIT编译缓存与运行时契约.md](<../guides/模块七-DeepSeek开放技术详解/6.5-DeepJIT编译缓存与运行时契约.md>) |
| D6.6 | [6.6-Ascend后端与跨硬件能力矩阵.md](<../guides/模块七-DeepSeek开放技术详解/6.6-Ascend后端与跨硬件能力矩阵.md>) |
| D7.1 | [7.1-3FS存储架构一致性与训练IO.md](<../guides/模块七-DeepSeek开放技术详解/7.1-3FS存储架构一致性与训练IO.md>) |
| D7.2 | [7.2-smallpond数据处理分区与恢复.md](<../guides/模块七-DeepSeek开放技术详解/7.2-smallpond数据处理分区与恢复.md>) |
| D7.3 | [7.3-FireFlyer集群与通信存储协同.md](<../guides/模块七-DeepSeek开放技术详解/7.3-FireFlyer集群与通信存储协同.md>) |
| D7.4 | [7.4-DSec沙箱与Agent训练环境Infra.md](<../guides/模块七-DeepSeek开放技术详解/7.4-DSec沙箱与Agent训练环境Infra.md>) |
| F3.9 | [3.9 Tokenization与词嵌入.md](<../guides/模块一-前置知识/transformer/3.9 Tokenization与词嵌入.md>) |
| P1.1 | [1.1-跨训练推理的Roofline与关键路径.md](<../guides/模块六-性能工程与生产实践/1.1-跨训练推理的Roofline与关键路径.md>) |
| P1.2 | [1.2-统计测量基准设计与性能归因.md](<../guides/模块六-性能工程与生产实践/1.2-统计测量基准设计与性能归因.md>) |
| P2.1 | [2.1-从GPU时间线到微架构瓶颈.md](<../guides/模块六-性能工程与生产实践/2.1-从GPU时间线到微架构瓶颈.md>) |
| P2.2 | [2.2-编译CUDA图与动态Shape工程.md](<../guides/模块六-性能工程与生产实践/2.2-编译CUDA图与动态Shape工程.md>) |
| P3.1 | [3.1-NCCL与RDMA网络性能工程.md](<../guides/模块六-性能工程与生产实践/3.1-NCCL与RDMA网络性能工程.md>) |
| P3.2 | [3.2-NUMA数据加载与异步IO流水.md](<../guides/模块六-性能工程与生产实践/3.2-NUMA数据加载与异步IO流水.md>) |
| P4.1 | [4.1-低精度Kernel正确性与发布门禁.md](<../guides/模块六-性能工程与生产实践/4.1-低精度Kernel正确性与发布门禁.md>) |
| P4.2 | [4.2-AIInfra性能项目与综合实验.md](<../guides/模块六-性能工程与生产实践/4.2-AIInfra性能项目与综合实验.md>) |
| R1.1 | [1.1-SFT与RLHF训练全景.md](<../guides/模块五-LLM-RL后训练/1.1-SFT与RLHF训练全景.md>) |
| R1.2 | [1.2-轨迹张量与策略梯度.md](<../guides/模块五-LLM-RL后训练/1.2-轨迹张量与策略梯度.md>) |
| R10.2 | [10.2-最小RL闭环与框架迁移.md](<../guides/模块五-LLM-RL后训练/10.2-最小RL闭环与框架迁移.md>) |
| R11.1 | [11.1-长轨迹Mask与分布式Loss归一化.md](<../guides/模块五-LLM-RL后训练/11.1-长轨迹Mask与分布式Loss归一化.md>) |
| R11.2 | [11.2-GSPO序列比率与离策略校正.md](<../guides/模块五-LLM-RL后训练/11.2-GSPO序列比率与离策略校正.md>) |
| R11.3 | [11.3-Agent环境吞吐与训练稳定性.md](<../guides/模块五-LLM-RL后训练/11.3-Agent环境吞吐与训练稳定性.md>) |
| R2.1 | [2.1-PPO目标与裁剪机制.md](<../guides/模块五-LLM-RL后训练/2.1-PPO目标与裁剪机制.md>) |
| R2.2 | [2.2-GAE与PPO一轮训练.md](<../guides/模块五-LLM-RL后训练/2.2-GAE与PPO一轮训练.md>) |
| R3.1 | [3.1-GRPO原理与组内优势.md](<../guides/模块五-LLM-RL后训练/3.1-GRPO原理与组内优势.md>) |
| R3.2 | [3.2-GRPO变体与训练稳定性.md](<../guides/模块五-LLM-RL后训练/3.2-GRPO变体与训练稳定性.md>) |
| R4.1 | [4.1-偏好数据与奖励模型.md](<../guides/模块五-LLM-RL后训练/4.1-偏好数据与奖励模型.md>) |
| R4.2 | [4.2-DPO推导与实践边界.md](<../guides/模块五-LLM-RL后训练/4.2-DPO推导与实践边界.md>) |
| R6.1 | [6.1-多轮轨迹与工具环境.md](<../guides/模块五-LLM-RL后训练/6.1-多轮轨迹与工具环境.md>) |
| R7.1 | [7.1-角色编排与端到端数据流.md](<../guides/模块五-LLM-RL后训练/7.1-角色编排与端到端数据流.md>) |
| R7.2 | [7.2-verl-ROLL与slime框架实践.md](<../guides/模块五-LLM-RL后训练/7.2-verl-ROLL与slime框架实践.md>) |
| R8.1 | [8.1-混合部署与权重同步.md](<../guides/模块五-LLM-RL后训练/8.1-混合部署与权重同步.md>) |
| R8.2 | [8.2-异步Rollout与吞吐优化.md](<../guides/模块五-LLM-RL后训练/8.2-异步Rollout与吞吐优化.md>) |
| T10.1 | [10.1-MoE路由与专家并行.md](<../guides/模块三-分布式训练/10.1-MoE路由与专家并行.md>) |
| T10.2 | [10.2-MoE负载均衡与通信优化.md](<../guides/模块三-分布式训练/10.2-MoE负载均衡与通信优化.md>) |
| T11.1 | [11.1-多维并行拓扑与Rank编排.md](<../guides/模块三-分布式训练/11.1-多维并行拓扑与Rank编排.md>) |
| T11.2 | [11.2-训练性能建模与可靠恢复.md](<../guides/模块三-分布式训练/11.2-训练性能建模与可靠恢复.md>) |
| T12.1 | [12.1-语料治理去重与数据混合.md](<../guides/模块三-分布式训练/12.1-语料治理去重与数据混合.md>) |
| T12.2 | [12.2-GPTDataset索引Packing与确定性恢复.md](<../guides/模块三-分布式训练/12.2-GPTDataset索引Packing与确定性恢复.md>) |
| T12.3 | [12.3-数据加载吞吐与存储协同.md](<../guides/模块三-分布式训练/12.3-数据加载吞吐与存储协同.md>) |
| T13.1 | [13.1-一次Megatron训练迭代的完整调用链.md](<../guides/模块三-分布式训练/13.1-一次Megatron训练迭代的完整调用链.md>) |
| T13.2 | [13.2-ModuleSpec模型构造与TransformerEngine.md](<../guides/模块三-分布式训练/13.2-ModuleSpec模型构造与TransformerEngine.md>) |
| T13.3 | [13.3-梯度Buffer分布式优化器与状态所有权.md](<../guides/模块三-分布式训练/13.3-梯度Buffer分布式优化器与状态所有权.md>) |
| T13.4 | [13.4-HybridModel与广义张量并行.md](<../guides/模块三-分布式训练/13.4-HybridModel与广义张量并行.md>) |
| T13.5 | [13.5-Megatron中MLA与MTP的源码Gate.md](<../guides/模块三-分布式训练/13.5-Megatron中MLA与MTP的源码Gate.md>) |
| T13.6 | [13.6-ProcessGroupCollection与动态混合CP.md](<../guides/模块三-分布式训练/13.6-ProcessGroupCollection与动态混合CP.md>) |
| T14.1 | [14.1-分布式Checkpoint提交与重分片.md](<../guides/模块三-分布式训练/14.1-分布式Checkpoint提交与重分片.md>) |
| T14.2 | [14.2-FailSlow静默错误与自动恢复.md](<../guides/模块三-分布式训练/14.2-FailSlow静默错误与自动恢复.md>) |
| T14.3 | [14.3-集群调度观测与故障演练.md](<../guides/模块三-分布式训练/14.3-集群调度观测与故障演练.md>) |
| T15.1 | [15.1-参数FLOPs与训练预算的完整推导.md](<../guides/模块三-分布式训练/15.1-参数FLOPs与训练预算的完整推导.md>) |
| T15.2 | [15.2-五类大模型配置与并行方案设计.md](<../guides/模块三-分布式训练/15.2-五类大模型配置与并行方案设计.md>) |
| T15.3 | [15.3-训练课程优化算法与数值稳定性.md](<../guides/模块三-分布式训练/15.3-训练课程优化算法与数值稳定性.md>) |
| T15.4 | [15.4-ScalingLaws超参数迁移与实验设计.md](<../guides/模块三-分布式训练/15.4-ScalingLaws超参数迁移与实验设计.md>) |
| T16.1 | [16.1-训练Infra源码追问与计算题.md](<../guides/模块三-分布式训练/16.1-训练Infra源码追问与计算题.md>) |
| T16.2 | [16.2-可复现训练项目与系统设计验收.md](<../guides/模块三-分布式训练/16.2-可复现训练项目与系统设计验收.md>) |
| T17.1 | [17.1-TorchTitan与原生PyTorch训练生态.md](<../guides/模块三-分布式训练/17.1-TorchTitan与原生PyTorch训练生态.md>) |
| T17.2 | [17.2-MaxText与JAX的分片和训练系统.md](<../guides/模块三-分布式训练/17.2-MaxText与JAX的分片和训练系统.md>) |
| T17.3 | [17.3-开放训练框架选型与集成契约.md](<../guides/模块三-分布式训练/17.3-开放训练框架选型与集成契约.md>) |
| T17.4 | [17.4-AMD-TPU-Trainium与异构训练边界.md](<../guides/模块三-分布式训练/17.4-AMD-TPU-Trainium与异构训练边界.md>) |
| T2.1 | [2.1 集合通信原语详解.md](<../guides/模块三-分布式训练/2.1 集合通信原语详解.md>) |
| T2.2 | [2.2-Ring逐轮推导与通信性能分析.md](<../guides/模块三-分布式训练/2.2-Ring逐轮推导与通信性能分析.md>) |
| T3.2 | [3.2-梯度累积与有效Batch更新.md](<../guides/模块三-分布式训练/3.2-梯度累积与有效Batch更新.md>) |
| T4.3 | [4.3-两进程DDP实验与正确性验收.md](<../guides/模块三-分布式训练/4.3-两进程DDP实验与正确性验收.md>) |
| T5.1 | [5.1-ZeRO分片与显存通信账本.md](<../guides/模块三-分布式训练/5.1-ZeRO分片与显存通信账本.md>) |
| T5.2 | [5.2-FSDP2与Offload工程实践.md](<../guides/模块三-分布式训练/5.2-FSDP2与Offload工程实践.md>) |
| T5.3 | [5.3-HSDP与多实例分布式优化器.md](<../guides/模块三-分布式训练/5.3-HSDP与多实例分布式优化器.md>) |
| T6.1 | [6.1-张量并行的前向反向与通信.md](<../guides/模块三-分布式训练/6.1-张量并行的前向反向与通信.md>) |
| T6.2 | [6.2-序列并行与并行线性层实践.md](<../guides/模块三-分布式训练/6.2-序列并行与并行线性层实践.md>) |
| T7.1 | [7.1-流水线调度与Bubble推导.md](<../guides/模块三-分布式训练/7.1-流水线调度与Bubble推导.md>) |
| T7.2 | [7.2-交错流水线与负载均衡.md](<../guides/模块三-分布式训练/7.2-交错流水线与负载均衡.md>) |
| T8.1 | [8.1-混合精度与优化器数值链路.md](<../guides/模块三-分布式训练/8.1-混合精度与优化器数值链路.md>) |
| T8.2 | [8.2-激活重计算与显存生命周期.md](<../guides/模块三-分布式训练/8.2-激活重计算与显存生命周期.md>) |
| T8.3 | [8.3-FP8与FP4训练的缩放数值与验收.md](<../guides/模块三-分布式训练/8.3-FP8与FP4训练的缩放数值与验收.md>) |
| T9.1 | [9.1-长序列训练与上下文并行.md](<../guides/模块三-分布式训练/9.1-长序列训练与上下文并行.md>) |
| T9.2 | [9.2-RingAttention与长序列工程实践.md](<../guides/模块三-分布式训练/9.2-RingAttention与长序列工程实践.md>) |
| T9.3 | [9.3-RoPE扩展与长上下文训练课程.md](<../guides/模块三-分布式训练/9.3-RoPE扩展与长上下文训练课程.md>) |
| T9.4 | [9.4-Packing混合长度与动态序列并行.md](<../guides/模块三-分布式训练/9.4-Packing混合长度与动态序列并行.md>) |
| T9.5 | [9.5-有效上下文评测与稀疏混合架构.md](<../guides/模块三-分布式训练/9.5-有效上下文评测与稀疏混合架构.md>) |
| T9.6 | [9.6-AutoSP与HexiSeq动态长序列调度.md](<../guides/模块三-分布式训练/9.6-AutoSP与HexiSeq动态长序列调度.md>) |

[T1.1]: <../guides/模块三-分布式训练/1.1 分布式训练总论.md>
[T2.1]: <../guides/模块三-分布式训练/2.1 集合通信原语详解.md>
[T2.2]: <../guides/模块三-分布式训练/2.2-Ring逐轮推导与通信性能分析.md>
[T3.1]: <../guides/模块三-分布式训练/3.1 优化器原理与显存开销分析.md>
[T3.2]: <../guides/模块三-分布式训练/3.2-梯度累积与有效Batch更新.md>
[T4.1]: <../guides/模块三-分布式训练/4.1 数据并行详解.md>
[T4.2]: <../guides/模块三-分布式训练/4.2 PyTorch 数据并行从原理到实战.md>
[T4.3]: <../guides/模块三-分布式训练/4.3-两进程DDP实验与正确性验收.md>
[T5.1]: <../guides/模块三-分布式训练/5.1-ZeRO分片与显存通信账本.md>
[T5.2]: <../guides/模块三-分布式训练/5.2-FSDP2与Offload工程实践.md>
[T6.1]: <../guides/模块三-分布式训练/6.1-张量并行的前向反向与通信.md>
[T6.2]: <../guides/模块三-分布式训练/6.2-序列并行与并行线性层实践.md>
[T7.1]: <../guides/模块三-分布式训练/7.1-流水线调度与Bubble推导.md>
[T7.2]: <../guides/模块三-分布式训练/7.2-交错流水线与负载均衡.md>
[T8.1]: <../guides/模块三-分布式训练/8.1-混合精度与优化器数值链路.md>
[T8.2]: <../guides/模块三-分布式训练/8.2-激活重计算与显存生命周期.md>
[T8.3]: <../guides/模块三-分布式训练/8.3-FP8与FP4训练的缩放数值与验收.md>
[T9.1]: <../guides/模块三-分布式训练/9.1-长序列训练与上下文并行.md>
[T9.2]: <../guides/模块三-分布式训练/9.2-RingAttention与长序列工程实践.md>
[T9.3]: <../guides/模块三-分布式训练/9.3-RoPE扩展与长上下文训练课程.md>
[T9.4]: <../guides/模块三-分布式训练/9.4-Packing混合长度与动态序列并行.md>
[T9.5]: <../guides/模块三-分布式训练/9.5-有效上下文评测与稀疏混合架构.md>
[T10.1]: <../guides/模块三-分布式训练/10.1-MoE路由与专家并行.md>
[T10.2]: <../guides/模块三-分布式训练/10.2-MoE负载均衡与通信优化.md>
[T11.1]: <../guides/模块三-分布式训练/11.1-多维并行拓扑与Rank编排.md>
[T11.2]: <../guides/模块三-分布式训练/11.2-训练性能建模与可靠恢复.md>
[T12.1]: <../guides/模块三-分布式训练/12.1-语料治理去重与数据混合.md>
[T12.2]: <../guides/模块三-分布式训练/12.2-GPTDataset索引Packing与确定性恢复.md>
[T12.3]: <../guides/模块三-分布式训练/12.3-数据加载吞吐与存储协同.md>
[T13.1]: <../guides/模块三-分布式训练/13.1-一次Megatron训练迭代的完整调用链.md>
[T13.2]: <../guides/模块三-分布式训练/13.2-ModuleSpec模型构造与TransformerEngine.md>
[T13.3]: <../guides/模块三-分布式训练/13.3-梯度Buffer分布式优化器与状态所有权.md>
[T13.4]: <../guides/模块三-分布式训练/13.4-HybridModel与广义张量并行.md>
[T14.1]: <../guides/模块三-分布式训练/14.1-分布式Checkpoint提交与重分片.md>
[T14.2]: <../guides/模块三-分布式训练/14.2-FailSlow静默错误与自动恢复.md>
[T14.3]: <../guides/模块三-分布式训练/14.3-集群调度观测与故障演练.md>
[T15.1]: <../guides/模块三-分布式训练/15.1-参数FLOPs与训练预算的完整推导.md>
[T15.2]: <../guides/模块三-分布式训练/15.2-五类大模型配置与并行方案设计.md>
[T15.3]: <../guides/模块三-分布式训练/15.3-训练课程优化算法与数值稳定性.md>
[T15.4]: <../guides/模块三-分布式训练/15.4-ScalingLaws超参数迁移与实验设计.md>
[T16.1]: <../guides/模块三-分布式训练/16.1-训练Infra源码追问与计算题.md>
[T16.2]: <../guides/模块三-分布式训练/16.2-可复现训练项目与系统设计验收.md>
[P1.1]: <../guides/模块六-性能工程与生产实践/1.1-跨训练推理的Roofline与关键路径.md>
[P1.2]: <../guides/模块六-性能工程与生产实践/1.2-统计测量基准设计与性能归因.md>
[P2.1]: <../guides/模块六-性能工程与生产实践/2.1-从GPU时间线到微架构瓶颈.md>
[P2.2]: <../guides/模块六-性能工程与生产实践/2.2-编译CUDA图与动态Shape工程.md>
[P3.1]: <../guides/模块六-性能工程与生产实践/3.1-NCCL与RDMA网络性能工程.md>
[P3.2]: <../guides/模块六-性能工程与生产实践/3.2-NUMA数据加载与异步IO流水.md>
[P4.1]: <../guides/模块六-性能工程与生产实践/4.1-低精度Kernel正确性与发布门禁.md>
[P4.2]: <../guides/模块六-性能工程与生产实践/4.2-AIInfra性能项目与综合实验.md>
[R1.1]: <../guides/模块五-LLM-RL后训练/1.1-SFT与RLHF训练全景.md>
[R1.2]: <../guides/模块五-LLM-RL后训练/1.2-轨迹张量与策略梯度.md>
[R2.1]: <../guides/模块五-LLM-RL后训练/2.1-PPO目标与裁剪机制.md>
[R2.2]: <../guides/模块五-LLM-RL后训练/2.2-GAE与PPO一轮训练.md>
[R3.1]: <../guides/模块五-LLM-RL后训练/3.1-GRPO原理与组内优势.md>
[R3.2]: <../guides/模块五-LLM-RL后训练/3.2-GRPO变体与训练稳定性.md>
[R4.1]: <../guides/模块五-LLM-RL后训练/4.1-偏好数据与奖励模型.md>
[R4.2]: <../guides/模块五-LLM-RL后训练/4.2-DPO推导与实践边界.md>
[R5.1]: <../guides/模块五-LLM-RL后训练/5.1-可验证奖励与数据流水线.md>
[R5.2]: <../guides/模块五-LLM-RL后训练/5.2-评估指标与训练诊断.md>
[R6.1]: <../guides/模块五-LLM-RL后训练/6.1-多轮轨迹与工具环境.md>
[R6.2]: <../guides/模块五-LLM-RL后训练/6.2-On-Policy蒸馏与教师反馈.md>
[R7.1]: <../guides/模块五-LLM-RL后训练/7.1-角色编排与端到端数据流.md>
[R7.2]: <../guides/模块五-LLM-RL后训练/7.2-verl-ROLL与slime框架实践.md>
[R8.1]: <../guides/模块五-LLM-RL后训练/8.1-混合部署与权重同步.md>
[R8.2]: <../guides/模块五-LLM-RL后训练/8.2-异步Rollout与吞吐优化.md>
[R9.1]: <../guides/模块五-LLM-RL后训练/9.1-Checkpoint与可复现训练.md>
[R9.2]: <../guides/模块五-LLM-RL后训练/9.2-RL训练故障定位实战.md>
[R10.1]: <../guides/模块五-LLM-RL后训练/10.1-后训练数值实验.md>
[R10.2]: <../guides/模块五-LLM-RL后训练/10.2-最小RL闭环与框架迁移.md>
[R11.1]: <../guides/模块五-LLM-RL后训练/11.1-长轨迹Mask与分布式Loss归一化.md>
[R11.2]: <../guides/模块五-LLM-RL后训练/11.2-GSPO序列比率与离策略校正.md>
[R11.3]: <../guides/模块五-LLM-RL后训练/11.3-Agent环境吞吐与训练稳定性.md>
[D1.1]: <../guides/模块七-DeepSeek开放技术详解/1.1-DeepSeek模型与开源项目演进.md>
[D1.2]: <../guides/模块七-DeepSeek开放技术详解/1.2-开源代码开放权重与论文披露边界.md>
[D2.1]: <../guides/模块七-DeepSeek开放技术详解/2.1-MLA低秩压缩解耦RoPE与矩阵吸收.md>
[D2.2]: <../guides/模块七-DeepSeek开放技术详解/2.2-DeepSeekMoE细粒度专家与共享专家.md>
[D2.3]: <../guides/模块七-DeepSeek开放技术详解/2.3-NSA与DSA稀疏注意力.md>
[D2.4]: <../guides/模块七-DeepSeek开放技术详解/2.4-V4混合注意力CSA与HCA.md>
[D2.5]: <../guides/模块七-DeepSeek开放技术详解/2.5-mHC残差流形约束与数值稳定性.md>
[D2.6]: <../guides/模块七-DeepSeek开放技术详解/2.6-V4点1架构更新与原生视觉.md>
[D2.7]: <../guides/模块七-DeepSeek开放技术详解/2.7-CED编解码解耦与PrefixReplay.md>
[D2.8]: <../guides/模块七-DeepSeek开放技术详解/2.8-Engram条件记忆与分层卸载.md>
[D2.9]: <../guides/模块七-DeepSeek开放技术详解/2.9-CSA2与分层稀疏索引.md>
[D3.1]: <../guides/模块七-DeepSeek开放技术详解/3.1-FP8细粒度缩放与训练数值链路.md>
[D3.2]: <../guides/模块七-DeepSeek开放技术详解/3.2-DualPipe与DualPipeV流水调度.md>
[D3.3]: <../guides/模块七-DeepSeek开放技术详解/3.3-Muon优化器与V4训练配方.md>
[D3.4]: <../guides/模块七-DeepSeek开放技术详解/3.4-MTP与DSpark训练目标.md>
[D4.1]: <../guides/模块七-DeepSeek开放技术详解/4.1-GRPO与R1Zero到R1的训练路线.md>
[D4.2]: <../guides/模块七-DeepSeek开放技术详解/4.2-V3点2与V4点1AgentRL和环境系统.md>
[D4.3]: <../guides/模块七-DeepSeek开放技术详解/4.3-DeepSeekMath与自验证推理.md>
[D4.4]: <../guides/模块七-DeepSeek开放技术详解/4.4-蒸馏合成数据与1017自反馈更新.md>
[D5.1]: <../guides/模块七-DeepSeek开放技术详解/5.1-FlashMLA稀疏算子与KV缓存格式.md>
[D5.2]: <../guides/模块七-DeepSeek开放技术详解/5.2-DeepInfer端到端源码与执行生命周期.md>
[D5.3]: <../guides/模块七-DeepSeek开放技术详解/5.3-DSpark投机验证与置信度调度.md>
[D6.1]: <../guides/模块七-DeepSeek开放技术详解/6.1-DeepGEMM低精度矩阵与MegaMoE.md>
[D6.2]: <../guides/模块七-DeepSeek开放技术详解/6.2-DeepEP通信布局与异步生命周期.md>
[D6.3]: <../guides/模块七-DeepSeek开放技术详解/6.3-EPLB与LPLB专家副本和动态均衡.md>
[D6.4]: <../guides/模块七-DeepSeek开放技术详解/6.4-DeepSelect稀疏索引与TopK算法.md>
[D6.5]: <../guides/模块七-DeepSeek开放技术详解/6.5-DeepJIT编译缓存与运行时契约.md>
[D6.6]: <../guides/模块七-DeepSeek开放技术详解/6.6-Ascend后端与跨硬件能力矩阵.md>
[D7.1]: <../guides/模块七-DeepSeek开放技术详解/7.1-3FS存储架构一致性与训练IO.md>
[D7.2]: <../guides/模块七-DeepSeek开放技术详解/7.2-smallpond数据处理分区与恢复.md>
[D7.3]: <../guides/模块七-DeepSeek开放技术详解/7.3-FireFlyer集群与通信存储协同.md>
[D7.4]: <../guides/模块七-DeepSeek开放技术详解/7.4-DSec沙箱与Agent训练环境Infra.md>
[D8.1]: <../guides/模块七-DeepSeek开放技术详解/8.1-DeepSeekVL与Janus多模态路线.md>
[D8.2]: <../guides/模块七-DeepSeek开放技术详解/8.2-DeepSeekOCR与OCR2视觉压缩.md>
[F3.9]: <../guides/模块一-前置知识/transformer/3.9 Tokenization与词嵌入.md>
[C-FA1]: <../guides/模块二-CUDA编程与算子优化/6.1-FlashAttention V1详解.md>
[C-FA2]: <../guides/模块二-CUDA编程与算子优化/6.2-FlashAttention V2详解.md>
[C-Triton]: <../guides/模块二-CUDA编程与算子优化/7.1-Triton编程框架.md>
[C-Compiler]: <../guides/模块二-CUDA编程与算子优化/第7章-AI编译器.md>
[T5.3]: <../guides/模块三-分布式训练/5.3-HSDP与多实例分布式优化器.md>
[T9.6]: <../guides/模块三-分布式训练/9.6-AutoSP与HexiSeq动态长序列调度.md>
[T13.5]: <../guides/模块三-分布式训练/13.5-Megatron中MLA与MTP的源码Gate.md>
[T13.6]: <../guides/模块三-分布式训练/13.6-ProcessGroupCollection与动态混合CP.md>
[T17.1]: <../guides/模块三-分布式训练/17.1-TorchTitan与原生PyTorch训练生态.md>
[T17.2]: <../guides/模块三-分布式训练/17.2-MaxText与JAX的分片和训练系统.md>
[T17.3]: <../guides/模块三-分布式训练/17.3-开放训练框架选型与集成契约.md>
[T17.4]: <../guides/模块三-分布式训练/17.4-AMD-TPU-Trainium与异构训练边界.md>
