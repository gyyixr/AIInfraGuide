---
title: "第8章：多模态与 Agent 开放生态"
description: "按视觉理解、图像生成、文档压缩、Agent执行与协议接入划分DeepSeek开放生态，连接模型能力与可重建的端到端系统"
pubDate: 2026-11-07
category: "deepseek-open"
order: 80
chapter: 8
tags: ["DeepSeek", "Infra", "开放生态", "源码核对"]
---

模型能看图、生成图、读文档或调用工具，背后对应不同结构与软件组件。
本章把DeepSeek的多模态研究与agent应用层分开讲清，再说明它们怎样连接。

核对日期：2026-11-07。
项目源码按固定commit阅读，不把历史README中的环境版本当成所有当前平台的通用安装方式。

## 1. 五个层次

| 层次 | 例子 | 负责的问题 |
|---|---|---|
| 多模态模型 | VL2、Janus、OCR2、V4.1 | 图文表示与生成 |
| 推理后端 | 对应模型执行实现 | 权重、KV、kernel与调度 |
| 协议适配 | deepseek-recipe | 消息、prompt、token与响应转换 |
| Agent运行 | deepseek-harness | 工具循环、上下文、事件日志 |
| 执行环境 | DSec等sandbox | 资源、隔离、状态与恢复 |

一个层次成功运行，不能证明其他层次已完整实现。

## 2. 阅读顺序

| 页面 | 学习目标 |
|---|---|
| 8.1 VL与Janus | 区分视觉理解、离散图像生成和flow路线 |
| 8.2 OCR与OCR2 | 理解有损视觉压缩与causal flow query |
| 8.3 Harness | 跟踪turn、step、tool和持久事件 |
| 8.4 Recipe | 正确连接API格式、模板、tokenizer和stream |

本章同时用张量shape和事件状态解释数据流。
模型侧看维度，系统侧看版本、调用ID与提交边界。

## 3. 一张图片的三种处理方式

VL类模型提取视觉语义后回答问题。
OCR类模型重点恢复文字、布局和结构。
Janus类生成路径可以根据条件产生图像code或连续图像状态。

三种方式的训练目标与输出空间不同。
不能因为都包含vision encoder就认为可以互换。

## 4. 视觉token数必须实际计算

图片可能被切成多个局部crop和全局thumbnail。
特征压缩之后，还可能加入行分隔和视图分隔符。

VL2的一个2×3局部网格例子得到1415个视觉相关位置。
OCR2的6个局部crop加全局视图对应1120个视觉特征。

这些数字属于各自配置，不能当整个生态统一标准。

## 5. 输出格式也是模型接口

文字词表token、图像code ID和连续视觉embedding不是同一种对象。
工具调用与普通assistant文字也有不同结构。

正确的模型执行必须保留这些边界。
将所有内容随意拼成字符串，可能破坏控制与位置语义。

## 6. 从模型能力到可用agent

```text
用户任务与图片
  -> 协议转换与prompt编码
  -> 模型生成文字/工具动作
  -> Harness解析和调度
  -> Sandbox执行
  -> 工具结果与截图
  -> 下一轮模型上下文
```

每一轮都需要记录发生了什么，而不仅是最后显示了什么。

## 7. Harness与Recipe不是同一个东西

Recipe负责转换和解析，不执行工具，也不计算模型概率。
Harness负责有状态agent循环，使用模型adapter与工具executor。

两者可以组合，但不能把Recipe称作RL训练配方，把Harness称作optimizer训练框架。

## 8. 公开源码阅读的两个入口

模型项目先读processor与 `prepare_inputs_embeds`，确认图片怎样进入主干。
系统项目先读接口和状态机，确认请求何时准备、何时提交、何时恢复。

之后再看kernel或并发优化。
没有语义与状态基线，很难判断性能修改是否正确。

## 9. 本章的小实验

token数量、CFG logits、OCR2 mask、context budget都可在没有大模型权重时验证。

这些实验让读者确认计算与协议直觉。
它们不代表真实图像质量、OCR准确率或agent任务成功率。

要验证能力，需要对应权重、输入数据、环境与评估流程。

## 10. 多模态评估需要更多维度

除了模型名，还应记录：

- 图片分辨率与crop策略。
- 视觉token与文字token数量。
- 输出模式与最大长度。
- sampler和guidance参数。
- 工具与网络权限。
- harness与模板版本。

同一模型不同配置可能得到显著不同的成本与质量。

## 11. 学习验收

读完应能回答：

- VL2的行分隔符为何影响上下文预算。
- Janus为何拆开理解与生成视觉编码。
- CFG为何组合logits后再softmax。
- OCR2视觉prefix与causal queries的mask如何画。
- 一个turn为何包含多次模型请求。
- 实时stream与持久事件有什么区别。
- Recipe为何保留special token范围。
- 图像占位为何不能永远只计1token。

每个专题都给出具体算例、源码入口与答案。

## 12. 官方来源

- [VL2固定仓库](https://github.com/deepseek-ai/DeepSeek-VL2/tree/ef9f91e2b6426536b83294c11742c27be66361b1)。
- [Janus固定仓库](https://github.com/deepseek-ai/Janus/tree/1daa72fa409002d40931bd7b36a9280362469ead)。
- [OCR2固定仓库](https://github.com/deepseek-ai/DeepSeek-OCR-2/tree/2f3699ebbb96fa8af32212e8c170f2cc28730fad)。
- [Harness固定仓库](https://github.com/deepseek-ai/deepseek-harness/tree/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159)。
- [Recipe固定仓库](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)。

这些项目提供不同深度的开放材料；训练数据、生产服务和硬件结果仍需按具体artifact与版本分别核对。
