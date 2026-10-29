# 分布式训练模块验证记录

本次在原模块基础上新增 17 篇详解、深化原有导读并修订已有专题，模块现为 11 章导读与 23 篇详解。保留原文件路径，重新统一章节和专题排序。

## 站点交付

| 检查 | 结果 |
|---|---|
| `npm run build` | Astro 0 errors、0 warnings，480 个静态页面构建并建立 Pagefind 索引 |
| `npm run check:distributed-site` | 34 篇源文档对应 35 个页面（含入口），3109 个站内链接/锚点通过，无 KaTeX 错误 |
| `npm run check:rl-site` | 原 RL 模块 30 篇文档、2315 个链接/锚点回归通过 |
| Chrome 逐页检查 | 全部 34 篇文章打开；17 张 Mermaid 图渲染成功；无公式错误、JS 异常或网络失败 |
| 搜索与导航 | FSDP2 搜索有结果，桌面与移动菜单可进入模块 |
| 响应布局 | 1440×1000、1024×768、390×844 的模块页无页面级横向溢出 |
| Markdown 内代码 | 49 个 Python 代码块语法检查通过；不把外部依赖片段标作完整执行 |

Astro 仍有 8 条提示，主要为已有未使用符号与内联脚本提示。中间构建日志、浏览器 JSON 和截图保存在本地 `.local/`，不提交。

## 数学与程序验证

### 标准库实验

`npm run test:distributed` 的 5 项检查通过：

1. 四 rank Ring 六轮消息和归约结果 `[600,604,608,612]`。
2. 不同 rank 数的 Ring 结果与直接求和一致。
3. 列并行/行并行前向和反向与完整矩阵一致。
4. 高分数输入的 online softmax 对不同分块方式给出相同结果。
5. “平均局部 attention 输出”的反例：错误值 2、正确全局输出 3。

`results/reference-mechanics.json` 保留具体输入、消息和中间状态。这些是单进程数学模拟，没有真实 TP/CP/GPU 通信。

### 两个真实 CPU/Gloo 进程

`ddp_lab.py` 与 `npm run lab:ddp` 入口均已执行，环境为 Python 3.12.13、PyTorch 2.13.0+cu132，所有张量位于 CPU，backend 为 Gloo。

验证 AllReduce、AllGather、ReduceScatter、All-to-All 的实际返回；在有效元素数 3 与 5 的两个数据分片上，执行 `no_sync` 梯度累积、全局 norm 裁剪与三步 AdamW，并对照单进程完整 batch oracle：

| 指标 | 结果 |
|---|---:|
| 最大梯度误差 | 1.1102230246251565e-16 |
| 最大模型参数误差 | 1.3877787807814457e-17 |
| 最大 Adam 状态误差 | 3.469446951953614e-18 |
| 错误局部均值的梯度偏差 | 0.062083333333333435 |
| 末尾跨 rank 参数差 | 0 |
| 单进程 oracle 内存恢复后的参数差 | 0 |

具体结果在 `results/reference-ddp.json`。恢复检查仅为确定输入下的内存 model/AdamW 状态恢复，没有磁盘提交、进程崩溃或分布式事务实验。

### 文中独立算例与交叉审阅

- TP 输入与权重梯度经有限差分核对，最大误差约 4.19e-10。
- GPipe/1F1B 时序、激活驻留和层划分枚举实际执行核验。
- BF16 参数与 Adam moments dtype、FP32 master 小更新、显存字节量实际核验。
- 带 Dropout 的 CPU activation checkpoint 对照输出、输入/参数梯度和 RNG 状态通过。
- 分块 softmax 输出 3.237556871078246，与完整 softmax 一致。
- MoE dispatch/combine、8 卡 rank 分组、64 卡候选布局的标准库例子执行通过。
- 交叉复核纠正了 9.2 自测中因果 Attention 的贡献集合：KV 梯度只需要汇总 mask 可见的 query，最后一块 KV 可以只收到本 rank 的贡献。

## 来源与证据边界

Megatron 相关主要源码固定到 `7d6895ac6a3ee9e16c53f573f363d2a52b88bf94`；部分章节使用明确标注版本的 Megatron Core 0.18.1、PyTorch 2.9 文档和原始论文。文档版本与本地 PyTorch 2.13.0 的实验版本分别标注，不将静态文档说成安装环境的完整行为验证。

本次没有执行 NCCL、多 GPU FSDP/TP/PP/CP/EP、真实大模型训练或多机容错。数学公式、CPU 实验和配置阅读帮助建立正确理解，但不能据此报告 GPU 吞吐、生产可靠性或模型质量。
