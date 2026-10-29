#!/usr/bin/env python3
"""CPU-only teaching lab: four contextual bandits, one response token each.

Standard library only. This validates arithmetic and state transitions, not an
LLM, inference engine, distributed backend, or held-out task generalization.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
from pathlib import Path


def softmax(logits):
    shifted = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(shifted) for x in shifted]


def group_advantages(rewards, eps=1e-8):
    if len(rewards) < 2:
        raise ValueError("group size must be at least 2")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + eps) for r in rewards]


def ppo_surrogate(ratio, advantage, clip=0.2):
    return min(ratio * advantage, max(1 - clip, min(1 + clip, ratio)) * advantage)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Arguments are response sequence log-probabilities, not probabilities."""
    z = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return max(0.0, -z) + math.log1p(math.exp(-abs(z)))


def masked_mean(values, mask):
    if len(values) != len(mask) or any(m not in (0, 1) for m in mask):
        raise ValueError("values and binary mask must have matching lengths")
    count = sum(mask)
    if count == 0:
        raise ValueError("empty response mask")
    # Select before reduction: padding may even contain NaN or huge sentinels.
    return sum(v for v, m in zip(values, mask) if m) / count


def gae(rewards, values, terminated, gamma=1.0, lam=0.95):
    """One trajectory only: values has T+1 entries; final value bootstraps a
    truncation. terminated=True suppresses both bootstrap and GAE recurrence.
    A collector must call separately for each trajectory/collection boundary.
    """
    if len(values) != len(rewards) + 1 or len(terminated) != len(rewards):
        raise ValueError("expected T rewards, T terminations, T+1 values")
    advantages = [0.0] * len(rewards)
    carry = 0.0
    for t in reversed(range(len(rewards))):
        continuation = 0.0 if terminated[t] else 1.0
        delta = rewards[t] + gamma * continuation * values[t + 1] - values[t]
        carry = delta + gamma * lam * continuation * carry
        advantages[t] = carry
    return advantages


def objective_and_gradient(logits, reference, batch, clip=0.2, beta=0.02):
    """Minimized GRPO-style clipped surrogate + exact categorical reverse KL.

    Every row has one action/token. All groups have equal size. The old policy,
    reference, rewards and advantages are fixed inputs. Full-vocabulary KL is
    intentional here; this is not a sampled k3 KL implementation.
    """
    probs = [softmax(row) for row in logits]
    gradient = [[0.0] * len(row) for row in logits]
    policy_loss = 0.0
    clipped = 0
    for row in batch:
        prompt, action, advantage = row["prompt"], row["action"], row["advantage"]
        p = probs[prompt]
        ratio = p[action] / row["old_probability"]
        raw = ratio * advantage
        bounded = max(1 - clip, min(1 + clip, ratio)) * advantage
        policy_loss -= min(raw, bounded) / len(batch)
        # In the clipped, flat region the policy term contributes no gradient.
        active = raw <= bounded
        clipped += int(not active)
        if active:
            for j in range(len(p)):
                gradient[prompt][j] -= (
                    advantage * ratio * (float(j == action) - p[j]) / len(batch)
                )
    total_kl = 0.0
    entropy = 0.0
    for i, (p, q) in enumerate(zip(probs, reference)):
        logs = [math.log(a / b) for a, b in zip(p, q)]
        kl = sum(a * b for a, b in zip(p, logs))
        total_kl += kl / len(probs)
        entropy -= sum(a * math.log(a) for a in p) / len(probs)
        for j in range(len(p)):
            gradient[i][j] += beta * p[j] * (logs[j] - kl) / len(probs)
    return policy_loss + beta * total_kl, gradient, {
        "policy_loss": policy_loss, "reference_kl": total_kl,
        "entropy": entropy, "active_clip_fraction": clipped / len(batch),
    }


class BanditTrainer:
    """Actor and rollout are separate Python copies with explicit version IDs."""

    def __init__(self, seed=7, group_size=8, learning_rate=0.8, epochs=2, beta=0.02):
        if group_size < 2 or epochs < 1 or learning_rate <= 0 or beta < 0:
            raise ValueError("invalid training configuration")
        self.config = dict(seed=seed, group_size=group_size,
                           learning_rate=learning_rate, epochs=epochs, beta=beta)
        self.rng = random.Random(seed)
        self.actor_logits = [[0.0] * 4 for _ in range(4)]
        self.rollout_logits = copy.deepcopy(self.actor_logits)
        self.reference = [softmax(row) for row in self.actor_logits]
        # For prompt i, the environment accepts action (i+1) mod 4.
        self.targets = [1, 2, 3, 0]
        self.actor_version = self.rollout_version = 0

    def expected_reward(self):
        return sum(softmax(row)[target] for row, target in
                   zip(self.actor_logits, self.targets)) / len(self.targets)

    def collect(self):
        if self.actor_version != self.rollout_version:
            raise RuntimeError("rollout weights have not been synchronized")
        batch, zero_groups = [], 0
        for prompt, row in enumerate(self.rollout_logits):
            probs = softmax(row)
            actions = self.rng.choices(range(4), weights=probs, k=self.config["group_size"])
            rewards = [float(action == self.targets[prompt]) for action in actions]
            advantages = group_advantages(rewards)
            zero_groups += int(max(rewards) == min(rewards))
            for action, reward, advantage in zip(actions, rewards, advantages):
                batch.append(dict(prompt=prompt, action=action, reward=reward,
                                  advantage=advantage, old_probability=probs[action],
                                  behavior_version=self.rollout_version))
        return batch, zero_groups / len(self.targets)

    def step(self):
        batch, zero_group_fraction = self.collect()
        old_version = self.rollout_version
        for _ in range(self.config["epochs"]):
            loss, gradient, metrics = objective_and_gradient(
                self.actor_logits, self.reference, batch, beta=self.config["beta"])
            for i, row in enumerate(gradient):
                for j, g in enumerate(row):
                    self.actor_logits[i][j] -= self.config["learning_rate"] * g
        # Publish exactly once, after all optimizer epochs for this batch.
        self.actor_version += 1
        self.rollout_logits = copy.deepcopy(self.actor_logits)
        self.rollout_version = self.actor_version
        return {
            "step": self.actor_version, "behavior_version": old_version,
            "published_version": self.rollout_version,
            "sampled_reward_before_update": sum(r["reward"] for r in batch) / len(batch),
            "expected_reward_after_update": self.expected_reward(),
            "zero_group_fraction": zero_group_fraction,
            # Diagnostics are evaluated before the final optimizer epoch.
            "loss_before_last_update": loss, **metrics,
        }

    def state_dict(self):
        return copy.deepcopy({
            "schema_version": 1, "config": self.config,
            "actor_logits": self.actor_logits, "rollout_logits": self.rollout_logits,
            "reference": self.reference, "actor_version": self.actor_version,
            "rollout_version": self.rollout_version, "rng_state": self.rng.getstate(),
        })

    @classmethod
    def from_state(cls, state):
        if state["schema_version"] != 1:
            raise ValueError("unsupported checkpoint schema")
        obj = cls(**state["config"])
        for key in ("actor_logits", "rollout_logits", "reference", "actor_version", "rollout_version"):
            setattr(obj, key, copy.deepcopy(state[key]))
        def tuple_tree(value):
            return tuple(tuple_tree(v) for v in value) if isinstance(value, (list, tuple)) else value
        obj.rng.setstate(tuple_tree(state["rng_state"]))
        return obj


def numerical_examples():
    return {
        "group_rewards": [0, 0, 1, 1],
        "group_advantages": group_advantages([0, 0, 1, 1]),
        "ppo_positive_clipped": ppo_surrogate(1.5, 2.0),
        "ppo_negative_clipped": ppo_surrogate(0.5, -2.0),
        "gae": gae([0, 0, 1], [0.2, 0.3, 0.4, 0], [False, False, True], lam=0.9),
        "dpo_at_reference": dpo_loss(-2, -3, -2, -3),
        "masked_mean": masked_mean([1, 3, 999], [1, 1, 0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80, help="additional rollout/update iterations")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save", type=Path, help="write a JSON checkpoint after the last synchronization")
    parser.add_argument("--resume", type=Path, help="restore config, RNG, actor, reference and rollout")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    trainer = BanditTrainer.from_state(json.loads(args.resume.read_text())) if args.resume else BanditTrainer(seed=args.seed)
    initial = trainer.expected_reward()
    history = [trainer.step() for _ in range(args.steps)]
    result = {
        "scope": "CPU contextual bandit; four prompts, four actions, one response token; not an LLM benchmark",
        "python": platform.python_version(), "config": trainer.config,
        "numerical_examples": numerical_examples(),
        "initial_expected_reward": initial,
        "final_expected_reward": trainer.expected_reward(), "history": history,
    }
    for path, content in ((args.output, result), (args.save, trainer.state_dict())):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(path)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
