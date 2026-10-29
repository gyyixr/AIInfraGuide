# 分布式训练实验

两层证据分别运行，不把数学模拟写成真实多 GPU 实测。

## 标准库数学实验

```bash
python3 -m unittest discover -s labs/distributed -p 'test_*.py' -v
python3 labs/distributed/mechanics_lab.py --output labs/distributed/results/latest-mechanics.json
```

也可用 `npm run test:distributed` 与 `npm run lab:distributed`。覆盖四 rank Ring 的六轮消息、两路 TP 前向/反向矩阵、在线 softmax 分块合并及局部均值反例。5 项检查通过；参考输出为 `results/reference-mechanics.json`。这些是单进程数学模拟，不执行网络/GPU通信。

## 两进程 CPU/Gloo

准备一个装有 PyTorch 和 Gloo 支持的 Python 环境后：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  labs/distributed/ddp_lab.py --output labs/distributed/results/latest-ddp.json
```

或者 `AIINFRA_PYTHON=/path/to/environment/bin/python npm run lab:ddp`，将路径替换为自己的解释器。默认 `npm run lab:ddp` 使用 `python3`；runner 不安装依赖。

此实验真实启动两个 CPU/Gloo 进程，检查 AllReduce、AllGather、ReduceScatter、All-to-All，以及 DDP `no_sync`、不等有效元素数 3/5 的梯度缩放、全局裁剪、AdamW 三步、跨 rank 参数一致性。单进程 oracle 的 model+optimizer 内存恢复也被检查，不属于磁盘或分布式事务恢复。

参考环境 Python 3.12.13 / PyTorch 2.13.0+cu132，但本实验所有 tensor 在 CPU 上。参考输出在 `results/reference-ddp.json`：正确梯度最大误差约 1.11e-16；错误的局部均值平均偏差约 0.06208。设备、数据和容差仅对应本教学任务。

站点对应第 2.2、3.2、4.3、6.2、9.2 及 11.2 节。没有执行 NCCL、多 GPU FSDP/TP/PP/CP/EP，不能用这里的结果推断其性能或完整模型质量。
