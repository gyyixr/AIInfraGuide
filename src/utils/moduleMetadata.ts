import type { CategorySlug } from './categories';

export interface ModuleResource {
  /** 资源类型标签，如「练习平台」 */
  type: string;
  /** 资源名称 */
  title: string;
  /** 外部链接 */
  href: string;
  /** 一句话说明 */
  description: string;
}

export interface ModuleMeta {
  /** 模块中文标题 */
  title: string;
  /** 简短描述 */
  description: string;
  /** 主题色（与 GuideSidebar 的 gradientMap 一致） */
  color: string;
  /** 学习建议 */
  tips: string[];
  /** 前置要求（简短描述） */
  prerequisites?: string;
  /** 延伸资源（站外链接） */
  resources?: ModuleResource[];
}

export const MODULE_METADATA: Partial<Record<CategorySlug, ModuleMeta>> = {
  'performance-analysis': {
    title: '模块六：性能工程与生产实践',
    description: '把 Roofline、请求排队、GPU 时间线、编译与算子选择、网络和存储放进同一个可测量的工程流程，建立从正确性到吞吐、尾延迟与成本的验证方法。',
    color: '#F97316',
    prerequisites: 'CUDA 基础、分布式训练与推理优化的基本数据流',
    tips: [
      '先固定工作负载、精度和计时边界，再比较性能数字',
      '从端到端关键路径定位到具体算子或通信，避免只优化局部峰值',
      '用数值 oracle、重复测量、质量护栏和回退条件评估每个优化',
    ],
  },
  'deepseek-open': {
    title: '模块七：DeepSeek 开放技术详解',
    description: '按官方论文、开放权重与固定源码梳理 DeepSeek 的模型结构、训练与后训练、推理引擎、算子通信、存储数据和 Agent 生态；从 MLA、MoE 到 V4.1、DeepInfer、DeepGEMM、DeepEP 与 3FS，逐项说明机制和复现边界。',
    color: '#2563EB',
    prerequisites: 'Transformer、混合精度、分布式训练、KV Cache 和 RL 后训练基础',
    tips: [
      '先读开放版图与版本证据，区分代码开源、权重开放和论文披露',
      '模型、训练、推理和底层算子沿同一条数据流交叉阅读',
      '新旧版本的缓存格式、硬件要求和接口差异不可混用',
      '性能数据注明报告环境，CPU 算例与真实大模型复现分别验收',
    ],
    resources: [
      { type: '官方项目', title: 'DeepSeek GitHub', href: 'https://github.com/deepseek-ai', description: '官方模型、算子、通信、推理与数据基础设施项目' },
      { type: '官方发布', title: 'DeepSeek 研究与发布', href: 'https://www.deepseek.com/en/news/', description: '模型发布与公开技术资料入口，具体证据版本见各篇正文' },
    ],
  },
  'llm-rl': {
    title: '模块五：LLM RL 后训练',
    description:
      '从 SFT、RLHF 与 RLVR 出发，推导 PPO、GRPO、DPO，深入奖励工程、多轮 Agent、在线蒸馏、训练推理协同和故障排查；通过数值实验、长轨迹归一化、GSPO 与离策略校正串联完整数据流。',
    color: '#E11D48',
    tips: [
      '先阅读第 1—3 章并手算一轮优势、概率比和裁剪，再进入框架实现',
      '第 4—6 章连接偏好数据、奖励评估、多轮环境与教师反馈',
      '第 7—9 章重点跟踪张量、进程、通信和权重版本，区分算法目标与系统实现',
      '第 10 章提供无需 GPU 或下载模型的实验；框架源码核对与大规模训练验证分别标明',
      '第 11 章从长轨迹 mask、全局 loss 分母和策略版本进入 Agent RL 的稳定性问题',
    ],
    prerequisites: 'Transformer、自回归生成、概率与梯度、PyTorch；系统章节建议先学习分布式训练和推理优化',
    resources: [
      { type: '论文', title: 'Proximal Policy Optimization', href: 'https://arxiv.org/abs/1707.06347', description: '理解采样策略、当前策略与裁剪目标之间的关系' },
      { type: '论文', title: 'DeepSeekMath / GRPO', href: 'https://arxiv.org/abs/2402.03300', description: '从组内相对奖励理解无需独立价值模型的策略优化' },
      { type: '官方文档', title: 'verl', href: 'https://verl.readthedocs.io/en/latest/', description: '对照算法配置、训练后端与 rollout 引擎阅读实现，具体版本见第 7 章' },
    ],
  },
  prerequisites: {
    title: '模块一：前置知识',
    description:
      '涵盖 GPU 架构、编程语言基础、数学基础、Transformer 架构、PyTorch 框架和集合通信等核心前置知识，为后续深入 AI Infra 打好坚实基础。',
    color: '#06B6D4',
    tips: [
      '建议按章节顺序学习，每章内容相对独立但有递进关系',
      '如果已有相关背景知识，可以跳过对应章节直接进入 CUDA 编程模块',
      'Transformer 架构和 PyTorch 框架是后续所有模块的基础，建议重点掌握',
    ],
  },
  'cuda-optimization': {
    title: '模块二：CUDA 编程与算子优化',
    description:
      '从 CUDA 编程入门到经典算子实现（Reduce、GEMM、Softmax、Attention），再到 AI 编译器和性能分析工具链，系统掌握 GPU 编程与算子优化技术。',
    color: '#F59E0B',
    tips: [
      '建议先完成前置知识模块中的 GPU 硬件概论',
      '每个算子实现章节都有配套代码，建议动手实践',
      '性能分析工具链是贯穿整个模块的重要工具，建议尽早熟悉',
    ],
    prerequisites: '前置知识模块（尤其是第5章 GPU 硬件概论）',
    resources: [
      {
        type: '练习平台',
        title: 'CUDA 手撕算子 · 面试练习',
        href: 'https://kkcocoa.github.io/cuda-operator-interview/',
        description:
          '覆盖访存、Reduce、Softmax、Attention、GEMM 等算子的分级手写练习，附提示、自检清单与浏览器内代码编辑器',
      },
      {
        type: '练习平台',
        title: 'LeetGPU',
        href: 'https://leetgpu.com/',
        description:
          'GPU 编程版的 LeetCode，按 easy/medium/hard 分级的算子题库，支持 CUDA、Triton、PyTorch 等多种写法，无需本地 GPU 即可提交运行，可查看 PTX/SASS 与性能排行',
      },
    ],
  },
  'distributed-training': {
    title: '模块三：分布式训练',
    description:
      '从显存、梯度与通信原语逐步深入 DDP、ZeRO/FSDP、TP/PP/CP、MoE、低精度与超长上下文，再贯通数据治理、Megatron 源码、可靠恢复、训练预算和跨硬件框架；用逐轮算例与 CPU/Gloo 实验检验关键不变量。',
    color: '#10B981',
    tips: [
      '先读 1—4 章；用 2.2 的 Ring 逐轮表和 3.2 的不等 batch 推导校准通信与梯度语义',
      '第 4.3 节可在两进程 CPU/Gloo 环境运行，无需 GPU 或下载模型',
      '第 5—9 章按参数、激活和序列的切分方式深入，配合矩阵、时序与精度算例',
      '第 10—11 章将专家路由、rank 编排、性能建模与可靠恢复串起来；数学模拟和多卡实测分别说明',
      '第 12—14 章跟踪数据索引、训练调用链、优化器所有权与 checkpoint 提交',
      '第 15—17 章用预算、源码追问和框架对照设计可验证的训练方案',
    ],
    prerequisites: '前置知识模块（尤其是第6章 集合通信基础）',
  },
  'inference-optimization': {
    title: '模块四：推理优化',
    description:
      '从推理与量化、投机解码深入分布式容量、MoE、PD 解耦、缓存传输和生产故障；沿 SGLang/vLLM 请求生命周期阅读源码，并通过 235 题、算例、项目验收与模拟面试复习。',
    color: '#8B5CF6',
    tips: [
      'LLM 推理基础章节帮助理解推理场景的独特挑战',
      '量化与投机解码需要同时验证质量、目标分布、实际 kernel 和端到端收益',
      '第 6—9 章重点跟踪并行容量、PD 成本、传输取消与生产故障边界',
      '端侧推理建议在真实设备上同时评估延迟、内存、功耗和精度',
      '第 11 章对照固定源码理解请求、状态、缓存和执行；第 12 章用逐题速查与项目练习检验理解',
    ],
    prerequisites: '前置知识模块 + CUDA 编程基础',
  },
};

/** category slug → URL 前缀映射 */
export const CATEGORY_URL_PREFIX: Partial<Record<CategorySlug, string>> = {
  prerequisites: '/prerequisites',
  'cuda-optimization': '/cuda',
  'distributed-training': '/distributed',
  'inference-optimization': '/inference',
  'llm-rl': '/rl',
  'performance-analysis': '/performance',
  'deepseek-open': '/deepseek',
};
