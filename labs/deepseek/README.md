# DeepSeek 机制实验

这些实验区分数学、状态、官方CPU规划和普通PyTorch GPU验证，不会下载模型或自动安装DeepSeek专用kernel。

## 标准库

```bash
npm run test:deepseek
npm run lab:deepseek
```

14项检查包含分块scale、MoE gate/梯度、expanded布局、TopK、Sinkhorn、页映射、FP4字节、sink/LSE、投机分布与选择偏差、prefix预算、缓存键和pending版本可见性。参考输出为 `results/numerics-reference.json`。

## PyTorch MLA

使用装有PyTorch的解释器：

```bash
python labs/deepseek/mla_torch.py --device cpu --output labs/deepseek/results/latest-mla-cpu.json
python labs/deepseek/mla_torch.py --device cuda --output labs/deepseek/results/latest-mla-cuda.json
```

本次参考环境：PyTorch 2.13.0+cu132、CPU与RTX3090/SM86、float64。完整K/V路径与吸收路径的输出和梯度误差小于1e-12；漏掉RoPE分支的负对照约0.03613。结果在 `mla-cpu-reference.json`、`mla-cuda-reference.json`。

它使用很小的合成张量和普通PyTorch算子，没有调用FlashMLA、DeepGEMM或DeepInfer，也没有加载DeepSeek权重。

## 官方EPLB规划

准备 `deepseek-ai/EPLB` 固定commit `d52c72d5b2f2fb4c41afbf8eb21366820239913d` 的 `eplb.py`：

```bash
python labs/deepseek/verify_eplb.py --source /path/to/pinned/EPLB/eplb.py --output labs/deepseek/results/latest-eplb.json
```

脚本在加载前检查规范化源码SHA-256，自己不下载代码。规划参数中的4 GPU只表示模拟拓扑，实际运行在CPU；结果 `eplb-reference.json` 为副本计数 `[4,2,1,1]`、预测每GPU负载 `[60,60,60,60]`，没有专家执行/迁移性能测量。

对应学习内容为模块七第9章。更高层的单算子、模型、RDMA和生产复现需按各专题列出的硬件/版本/数据契约单独验收。
