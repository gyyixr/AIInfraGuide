---
title: "第11章：SGLang 与 vLLM 源码数据流"
description: "沿请求、token、页面、grammar分支和多模态特征追踪固定版本主链，明确执行与回收不变量"
pubDate: 2026-11-07
category: "inference-optimization"
chapter: 11
order: 110
tags: ["源码", "SGLang", "vLLM", "请求生命周期", "KV"]
---

## 本章的阅读主线

从一条文本请求出发，追踪它怎样变成token、怎样进入batch、怎样占用KV、怎样执行并返回。
最后检查取消、抢占和异步完成之后资源由谁回收。
理解这条链，比背诵类名更能帮助定位实际问题。

本章不承诺全仓覆盖。
每篇末尾给出重点读取的文件、行范围和支撑结论。
没有读取的分支不写成已验证能力。

## 固定源码基线

| 项目 | Commit |
|---|---|
| SGLang | `46d9427b913b20ce0d72b95a9209717448b37368` |
| vLLM | `9c8e90eb2637a863ca14e47fd436b10ed7ba6536` |

本地HEAD与涉及文件的工作区状态已经只读核对。
这是参考资料对应的snapshot，不是最新版本声明。
公共API、backend、类名和路径升级后都需要重新查验。

## 学习路径

| 小节 | 主线 | 要回答的问题 |
|---|---|---|
| [11.1 请求入口、Tokenizer 与多进程协议](./11.1-请求入口Tokenizer与多进程协议.md) | HTTP→engine request→stream | 输入和输出由谁持有 |
| [11.2 Scheduler 状态机与预算不变量](./11.2-Scheduler状态机与预算不变量.md) | waiting→running→commit | 本轮能做多少工作 |
| [11.3 KV 缓存索引、驱逐与事件协议](./11.3-KV缓存索引驱逐与事件协议.md) | semantic key→physical page | 哪些KV可共享和回收 |
| [11.4 ModelRunner、算子派发与Day-0接入](./11.4-ModelRunner算子派发与DayZero接入.md) | schedule→tensor→kernel | 实际执行了什么 |
| [11.5 结构化输出与投机分支状态机](./11.5-结构化输出与投机分支状态机.md) | compile→mask→verify→commit | 候选grammar状态怎样回滚 |
| [11.6 多模态处理、Encoder与缓存生命周期](./11.6-多模态处理Encoder与缓存生命周期.md) | processor→encoder→KV→retire | 视觉位置与缓存由谁持有 |

建议先掌握第1—2章基础与第6—9章资源、协议和性能模型。
源码阅读不要求先从文件第一行读到最后一行。
先找边界，再按问题深入核心函数和异常路径。

## 1. 同时跟踪四种东西

```text
request identity
  request_id / attempt / generation

token progress
  prompt / computed / accepted output / speculative

memory ownership
  block table / cache refs / inflight access

completion
  model result / transfer completion / stream finish
```

它们不是同一个done状态。
HTTP结束后可能仍有DMA；token已生成不代表已写入下一轮KV。
区分这些状态，是理解取消和抢占的前提。

## 2. 两个框架的职责对齐

SGLang固定主链使用TokenizerManager、scheduler Req/Batch、worker与detokenizer回传。
vLLM V1固定主链区分AsyncLLM、EngineCore、scheduler Request、OutputProcessor和collector。

| 业务能力 | 阅读焦点 |
|---|---|
| 输入 | normalize、render、tokenize、validate |
| 调度 | token/sequence/KV预算、fairness |
| 缓存 | identity、longest prefix、refs、eviction |
| 执行 | position、slot、metadata、backend |
| 输出 | sample、stop、delta、usage |
| 结束 | abort、stale、drain、free |

表格按职责比较，不表示两个框架的状态机同构。
类名相似也不能证明同一语义。

## 3. 本章必须掌握的细节

中间prefill chunk不能提交伪生成token。
Filter batch必须同时筛请求与相应tensor元数据。
抢占可以重建KV，但不能重新采样已经接受的输出。
旧结果能否交付与能否修改新计数是两个问题。

Prefix命中需要连续语义身份，free queue可能仍保存有效缓存。
Cache events是外部提示，不是物理页面所有权证明。
ModelRunner零token计划可能仍有connector或分布式协议工作。
显式指定不合法backend与自动选择候选具有不同错误语义。
Grammar编译上下文、请求matcher和临时spec状态不能混为一个共享对象。
多模态processor缓存、encoder结果与decoder KV各有独立身份和回收边界。

## 4. 为什么需要CPU教学模型

源码很大，直接运行完整框架难以观察每个状态交错。
每篇附带只依赖标准库的小模型，验证一个窄不变量。
例如注册失败回滚、页预算、父hash链和显式backend拒绝。

这些模型不是框架重实现。
它们不能证明GPU memory ordering、真实IPC性能或多机恢复。
验证范围在代码前后明确说明。

## 5. 阅读与验证步骤

1. 固定commit并检查目标文件dirty状态。
2. 找到入口symbol，读取真实函数而非只看搜索结果。
3. 写出输入/输出表示、owner和完成事件。
4. 追一条正常请求。
5. 追一条异常、取消或抢占请求。
6. 将核心不变量写成小算例。
7. 保留未覆盖分支和运行限制。

升级时按symbol与behavior contract重新核对。
行号只属于当前固定引用，不是稳定API。

## 6. 学习验收

- 画出文本到stream的进程与数据边界。
- 手算混合batch的token数、positions与新增页。
- 解释chunk、preemption、stale output的区别。
- 描述cache插入失败时资源如何回滚。
- 找出实际backend选择与错误路径。
- 给新模型写出从reference到canary的接入矩阵。
- 从schema到mask解释正式历史与临时候选的区别。
- 手算视觉token、encoder features与decoder KV，并追踪共享引用退出。

下一章通过40道综合题与八类项目练习这些能力。

## 7. 参考映射

主要对应参考宝典FND-001、SCH-001—020、KV-001—020、KERN相关源码与项目阅读方法。
涉及生产协议与性能结论时，回到第7—9章的明确口径。

[SGLang固定源码](https://github.com/sgl-project/sglang/tree/46d9427b913b20ce0d72b95a9209717448b37368/python/sglang/srt)与[vLLM V1固定源码](https://github.com/vllm-project/vllm/tree/9c8e90eb2637a863ca14e47fd436b10ed7ba6536/vllm/v1)提供可追溯入口。
本章没有把snapshot静态阅读写成实际引擎性能验证。
