---
title: "第5章：推理算子与 DeepInfer 运行时"
description: "从FlashMLA缓存与索引契约，到DeepInfer请求生命周期和DSpark验证调度，连接张量正确性、状态所有权与服务性能"
pubDate: 2026-11-07
category: "deepseek-open"
order: 50
chapter: 5
tags: ["DeepSeek", "Infra", "开放生态", "源码核对"]
---

第2章定义模型怎样访问历史，第3—4章解释这些能力怎样训练。
本章进入推理执行：一个请求如何变成张量、经过kernel与采样，再把可提交的token返回调用者。

核对日期：2026-11-07。
主要对象为当前FlashMLA、DeepInfer公开离线核心，以及DSpark论文和DeepSpec实现。

## 1. 一条请求的机制地图

```text
请求与模型配置
  -> scheduler选择序列和物理状态
  -> 生成query与稀疏KV索引
  -> FlashMLA计算attention
  -> 模型logits与DSpark草稿
  -> target验证、stop与grammar处理
  -> 提交输出、更新状态、释放资源
```

计算正确与生命周期正确缺一不可。
一个token数值正确，如果属于错误请求或过早提交，整体结果仍然错误。

## 2. 三个专题的阅读顺序

| 页面 | 核心问题 | 先做的数例 |
|---|---|---|
| 5.1 FlashMLA | 索引、缓存格式与返回统计量怎样解释 | 逻辑页到物理页、528/288字节 |
| 5.2 DeepInfer | 请求、worker与输出对象何时有效 | 并行度推导和最后一轮flush |
| 5.3 DSpark | 如何保留目标分布并合理分配验证预算 | 拒绝采样、前缀存活和选择偏差 |

读5.1前可回看2.1与2.9。
读5.3前可回看3.4的draft与confidence训练目标。

## 3. 四种地址与计数不要混用

| 对象 | 说明 |
|---|---|
| 逻辑token位置 | 请求中的第几个token |
| 物理KV索引 | 分配到哪一页、页内哪个位置 |
| 候选top-k位置 | 当前query实际访问哪些KV |
| 已提交输出数 | 用户或调用者可以依赖的结果长度 |

候选已经在GPU算出，不代表已经通过验证。
序列从CPU队列移除，也不代表设备不再引用其buffer。

## 4. 记录字节数与模型平均字节数

FlashMLA当前格式中，一条512维FP8记录为528字节。
对应FP4记录为288字节，但还受具体cache参数位置约束。

V4.1的890 bytes/token是结合跨层共享、序列压缩与index K后的全局账本。
它不等于每层cache的末维，也不能再乘40层。

5.1解释物理记录，2.9解释跨模型累计口径。

## 5. 特殊边界必须先验证

全无效稀疏行、空输出、EOS、取消与最后一轮结果，常比普通随机输入更容易暴露错误。

例如当前稀疏prefill以零输出和特定LSE sentinel处理全无效行。
不能直接用普通空集合log-sum-exp假设替代实现契约。

先用小确定数组检查边界，再启用融合与图执行。

## 6. DeepInfer的三个生命周期

请求生命周期决定进入、推进、完成与取消。
张量生命周期决定哪一步读写哪些KV与metadata。
输出对象生命周期决定pool中的结果何时必须clone。

ZOS把部分后处理推迟到下一轮，增加重叠机会。
因此最后一次decode后仍可能需要flush，而不是直接退出循环。

## 7. 并行维度也有语义

本快照的world size由CP×DP与Engram并行范围共同决定。
EP是兼容网格中的专家通信视图，不能再无条件乘一次。

5.2的两个八卡例子说明同一个设备集合如何承担不同并行角色。
实际部署还需要核对rank、设备绑定和通信组成员。

## 8. DSpark的两个正确性层次

基础拒绝采样用真实proposal概率和残差分布恢复target分布。
验证长度调度还要避免依赖当前候选的未来信息而产生选择偏差。

同步early-stop方案与使用历史预测决定容量的ZOS方案应分别理解。
吞吐优化不能跳过采样语义证明。

## 9. 一个接受长度数例

条件接受率为 `[0.9,0.8,0.5]`。
前缀存活概率为 `[0.9,0.72,0.36]`。

预计接受草稿数为1.98。
不是把三个条件接受率直接相加得到2.2。

后面的候选必须先满足前面全部接受这一条件。

## 10. 三个检查问题与答案

**问题：固定DeepInfer主仓库SHA是否足够定义依赖？**

答案：还需保存vendor gitlink；各库独立最新HEAD不一定是该引擎验证过的组合。

**问题：保留SWA窗口加回退预算，是否证明bounded replay精确？**

答案：这是资源范围要求；回放截断本身仍有近似，需单独评估质量。

**问题：高接受率是否保证高吞吐？**

答案：还要计算draft、验证、调度与负载成本，低价值后缀可能挤占其他请求容量。

## 11. 实验与性能证据边界

本章包含固定源码阅读、数学推导和小规模oracle。
未把CPU例子称为FlashMLA GPU/NPU实测，也未加载完整V4.1运行生产服务。

性能评估应固定模型、cache命中、并发、输入输出长度、采样、硬件与依赖版本。
离线核心吞吐与在线SLO下的goodput需要分别测量。

## 12. 官方来源与后续连接

- [FlashMLA固定README](https://github.com/deepseek-ai/FlashMLA/blob/395cb3afbb331fef7b1b21e9340861b9436b42d8/README.md)。
- [DeepInfer固定README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)。
- [DSpark论文](https://arxiv.org/html/2607.05147v1)。
- [DeepSpec固定实现](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2)。

第6章继续拆解推理核心依赖的GEMM、通信、负载规划、TopK与编译层。
