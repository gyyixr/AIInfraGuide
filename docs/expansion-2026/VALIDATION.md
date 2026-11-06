# 2026-11-07 内容扩充与本地交付验收

本次从 `3ad88e4` 开始扩充，保留此前同步上游和 fork 自有内容的结果。资料与实验的证据强度分别记录，网页可访问不代表所有硬件实验已经执行。

## 1. 内容与来源

- 三份指定材料的 SHA-256 在收尾时重新计算，与 [reference-manifest.json](./reference-manifest.json) 全部一致。
- 原有 326 份已跟踪 `docs/` 文件全部保留；新增 125 篇教程/学习导航。主要参考章节的承接位置见 [训练审计](./TRAINING_COVERAGE_AUDIT.md)、[推理审计](./INFERENCE_COVERAGE_AUDIT.md) 和 [DeepSeek 审计](./DEEPSEEK_COVERAGE_AUDIT.md)。
- 推理宝典 235 个题号、题干、顺序及逐题入口保留，另附专题和模拟面试；训练报告按机制重组，并未声称原报告所有题目已逐题迁移。
- DeepSeek 新增 9 章、39 篇详解和 9 篇导读，覆盖架构、训练、RL、推理、算子通信、存储/数据、跨硬件和 Agent 生态。当前与历史版本、代码与权重许可、论文披露与完整实现分别说明。
- 固定源码、模型 revision、论文及实际读取范围见本目录各 `*-source-snapshots.json` 和 `deepseek-architecture-sources.json`。研究前沿和未执行的完整项目保留在覆盖矩阵中，不把主要章节映射解释成所有研究细节均已实现。

## 2. 可运行代码与复核

本轮收尾执行的三个标准库测试套件全部通过：RL 9 项、分布式数学 5 项、DeepSeek 14 项，共 28 项。

```bash
npm run test:rl
npm run test:distributed
npm run test:deepseek
```

正文中的独立 CPU 示例在各扩写批次实际执行，日志归档到 [evidence/](./evidence/)。包括训练索引/成本/恢复、长轨迹与 GSPO、请求/KV/队列状态、性能模型、语法回滚、多模态缓存和数值模型。日志中的重复复核不累计成新的独立实验。框架调用片段、示意控制流和目标硬件代码并未自动执行。

- DeepSeek 系统篇 13 个 Python 块通过语法解析，其中 11 个标准库程序执行成功；其余两个为 Smallpond 框架示例和 DeepEP 示意控制流，明确未运行。见 [系统示例日志](./evidence/deepseek-systems-examples.json)。
- 新训练核心补齐包 18 个 CPU 程序通过；[日志](./evidence/training-final-gap-fill-oracles.json)记录实际输出。
- FA3/FA4、语法约束、多模态三篇补充的 3 个 CPU oracle 通过；[日志](./evidence/aiinfraguide-inference-gap-closure-validation.json)记录检查范围。
- 架构与系统篇分别交叉审阅，修正算术强度上下界、JIT 摘要缓存寿命、DSpark 历史置信度、FlashMLA 后端能力和模型参考实现边界。审阅记录及作者示例在 `evidence/` 中保留。

## 3. 实际硬件证据

[MLA CPU 结果](../../labs/deepseek/results/mla-cpu-reference.json)和 [MLA CUDA 结果](../../labs/deepseek/results/mla-cuda-reference.json)来自 PyTorch 2.13.0+cu132、float64 小合成张量。CUDA 设备为 RTX 3090 / SM86：输出最大绝对误差约 `5.55e-17`，梯度最大误差约 `2.60e-18`，漏掉 RoPE 的负对照误差约 `0.03613`。

这是普通 PyTorch MLA 矩阵吸收的前向/反向实验，未调用 FlashMLA/DeepGEMM 专用 kernel，也未加载 DeepSeek 权重。

[EPLB 结果](../../labs/deepseek/results/eplb-reference.json)实际执行固定官方 Python 规划代码，CPU 上得到副本数 `[4,2,1,1]`、预测 GPU 负载 `[60,60,60,60]`。四 GPU 是规划拓扑，未发生真实专家迁移或 RDMA 通信。

此前两进程 CPU/Gloo DDP 结果保留于 `labs/distributed/`；本次未将历史实跑重新描述成新增 GPU/集群运行。未重训全模型、未部署 3FS/DSec/HAI、未执行 Ascend 或多机生产压测。

## 4. 网站与浏览器

最终 `npm run build` 通过：Astro check 0错误、0警告，构建及Pagefind索引均为730页。`npm run check:content`通过七模块与全站检查：164281次本地链接、3808次本地资源引用，0失效目标/锚点/KaTeX错误。730个页面的HTTP响应均为200，并与当前构建HTML逐字节一致。

浏览器复核28个代表页面和33个响应式场景，0公式/图片/JS/网络错误、0横向页面溢出；题库文章链接和可见搜索结果实际点击成功。

最终构建和检查结果以 [site-link-validation.json](./site-link-validation.json)、[browser-validation.json](./browser-validation.json) 和 [http-validation.json](./http-validation.json) 为准。

```bash
npm run build
npm run check:content
python3 scripts/check-built-site.py --output docs/expansion-2026/site-link-validation.json
```

检查覆盖七模块的源文件/章节/页面数量、全站本地链接和锚点、图片/脚本/样式引用与 KaTeX 错误；外部 URL 计数但不伪装成已验证可访问。浏览器检查目录、代表文章、235 题跳转、可见搜索结果、Mermaid、公式及桌面/移动布局。历史文章和更新日志的坏链接在本轮一并修复。

本地预览地址为 `http://localhost:4321/AIInfraGuide/`。Windows `curl.exe` 直连 localhost 与 `127.0.0.1` 均返回 HTTP 200，见 [主机检查](./evidence/windows-local-access.json)。PowerShell HttpWebRequest 在本环境出现超时，未据此修改网络配置或推断所有客户端均不可访问。

预览服务关闭后可重新运行 `npm run site:local`；持续编辑时使用 `npm run dev:local`。学习正文在浏览器阅读，实验运行需要终端；外部论文/仓库、远程图片及 CDN 资源仍需要网络，不把本地部署描述成完全离线归档。
