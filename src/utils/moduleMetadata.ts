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
  'llm-rl': {
    title: '模块五：LLM RL 后训练',
    description:
      '从 SFT、RLHF 与 RLVR 的训练全景出发，逐步推导 PPO、GRPO、DPO，深入奖励工程、多轮 Agent、在线蒸馏、训练推理协同和故障排查，最后用可运行的数值实验串联完整数据流。',
    color: '#E11D48',
    tips: [
      '先阅读第 1—3 章并手算一轮优势、概率比和裁剪，再进入框架实现',
      '第 4—6 章连接偏好数据、奖励评估、多轮环境与教师反馈',
      '第 7—9 章重点跟踪张量、进程、通信和权重版本，区分算法目标与系统实现',
      '第 10 章提供无需 GPU 或下载模型的实验；框架源码核对与大规模训练验证分别标明',
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
      '从显存账本与通信原语出发，推导梯度累积、DDP、ZeRO/FSDP2、TP/SP、流水线、混合精度、CP、MoE 与多维拓扑，通过逐轮算例、源码入口和真实两进程 CPU/Gloo 实验建立可复查的训练系统知识。',
    color: '#10B981',
    tips: [
      '先读 1—4 章；用 2.2 的 Ring 逐轮表和 3.2 的不等 batch 推导校准通信与梯度语义',
      '第 4.3 节可在两进程 CPU/Gloo 环境运行，无需 GPU 或下载模型',
      '第 5—9 章按参数、激活和序列的切分方式深入，配合矩阵、时序与精度算例',
      '第 10—11 章将专家路由、rank 编排、性能建模与可靠恢复串起来；数学模拟和多卡实测分别说明',
    ],
    prerequisites: '前置知识模块（尤其是第6章 集合通信基础）',
  },
  'inference-optimization': {
    title: '模块四：推理优化',
    description:
      '覆盖 LLM 推理基础、推理引擎核心技术、主流推理框架、量化、Speculative Decoding、PD 解耦架构、端侧推理，以及性能分析与端到端实战。',
    color: '#8B5CF6',
    tips: [
      'LLM 推理基础章节帮助理解推理场景的独特挑战',
      '量化和 Speculative Decoding 是当前最热门的推理优化方向',
      '建议结合主流推理框架（如 vLLM）进行实际部署练习',
      '端侧推理建议在真实设备上同时评估延迟、内存、功耗和精度',
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
};
