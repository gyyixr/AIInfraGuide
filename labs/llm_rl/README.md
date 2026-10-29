# LLM RL 后训练数值实验

配套模块五第 10 章。Python 3.10+，仅使用标准库，无需 GPU、模型下载或第三方包。

```bash
# 从仓库根目录执行
python3 -m unittest discover -s labs/llm_rl -p 'test_*.py' -v
python3 labs/llm_rl/rl_lab.py --steps 80 --output labs/llm_rl/results/latest.json
```

`npm run test:rl` / `npm run lab:rl` 是同样的快捷入口。

训练对象是 4 个上下文 × 4 个动作的 logits 表；每个回答只含 1 个动作。真实执行采样、规则奖励、组内优势、PPO 裁剪、精确分类分布 KL、两次梯度更新和显式权重发布。它用于理解 GRPO 风格数据流，不是 LLM 或推理框架 benchmark。

`results/reference-run.json` 是 Python 3.12.3、seed=7 的实测记录：80 轮后精确期望奖励从 0.25 升至 0.9885796050166068。此值只衡量四个训练上下文，不代表泛化。

9 项检查覆盖组归一化/零方差、裁剪符号、GAE 终止/截断、DPO 数值稳定性、padding mask、16 个参数的有限差分、学习方向及 reference 冻结、精确恢复、未同步版本拒绝。

断点恢复：

```bash
python3 labs/llm_rl/rl_lab.py --steps 40 --save .local/rl-checkpoint.json --output .local/rl-first.json
python3 labs/llm_rl/rl_lab.py --steps 40 --resume .local/rl-checkpoint.json --output .local/rl-resumed.json
python3 labs/llm_rl/rl_lab.py --steps 80 --output .local/rl-continuous.json
```

比较 `rl-continuous.json` 中 `history[40:]` 与 `rl-resumed.json` 的 `history`，在相同环境下应完全一致。恢复包含配置、actor/rollout/reference、版本和 RNG 内部状态；这里使用无动量梯度下降，没有优化器动量。文件使用同目录临时写入后替换，不提供多机事务或掉电持久性保证。

日志：`sampled_reward_before_update` 来自旧策略样本；`expected_reward_after_update` 来自更新后完整动作分布；loss/KL/entropy/clip 在最后一次梯度更新前测量。`active_clip_fraction` 表示 surrogate 真正选中平坦分支的比例，不等同于所有 ratio 超界的比例。

延伸到真实 LLM 时必须新增 token shift、response mask、EOS、采样分布、独立验证集和训练/推理后端一致性检查。完整 GPU、多机框架训练未在本实验中验证。
