"""Numerical/semantic checks: gradient oracle, masking, versioning and recovery."""
import copy
import json
import math
import unittest

from rl_lab import (BanditTrainer, dpo_loss, gae, group_advantages,
                    masked_mean, objective_and_gradient, ppo_surrogate)


class RLNumericsTest(unittest.TestCase):
    def test_group_normalization_and_zero_variance(self):
        a = group_advantages([0, 0, 1, 1])
        for got, expected in zip(a, [-1, -1, 1, 1]):
            self.assertAlmostEqual(got, expected, places=6)
        self.assertEqual(group_advantages([1, 1, 1, 1]), [0, 0, 0, 0])
        with self.assertRaises(ValueError):
            group_advantages([1])

    def test_clipping_is_sign_dependent(self):
        self.assertAlmostEqual(ppo_surrogate(1.5, 2), 2.4)
        self.assertAlmostEqual(ppo_surrogate(0.5, -2), -1.6)
        self.assertAlmostEqual(ppo_surrogate(0.5, 2), 1.0)
        self.assertAlmostEqual(ppo_surrogate(1.5, -2), -3.0)

    def test_gae_termination_and_truncation(self):
        got = gae([0, 0, 1], [0.2, 0.3, 0.4, 0], [False, False, True], lam=0.9)
        for value, expected in zip(got, [0.676, 0.64, 0.6]):
            self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(gae([0], [0.3, 0.8], [False])[0], 0.5)
        self.assertAlmostEqual(gae([0], [0.3, 0.8], [True])[0], -0.3)

    def test_dpo_direction_and_numeric_stability(self):
        base = dpo_loss(-2, -3, -2, -3)
        self.assertAlmostEqual(base, math.log(2))
        self.assertLess(dpo_loss(-1, -4, -2, -3), base)
        self.assertTrue(math.isfinite(dpo_loss(-10000, 0, 0, 0)))

    def test_padding_cannot_change_loss(self):
        self.assertEqual(masked_mean([1, 3, float("nan")], [1, 1, 0]), 2)
        with self.assertRaises(ValueError):
            masked_mean([1, 2], [0, 0])

    def test_gradient_matches_finite_differences(self):
        trainer = BanditTrainer()
        batch, _ = trainer.collect()
        logits = [[0.3, -0.2, 0.1, -0.4] for _ in range(4)]
        _, gradient, _ = objective_and_gradient(logits, trainer.reference, batch, beta=0.13)
        h = 1e-6
        for i in range(4):
            for j in range(4):
                plus, minus = copy.deepcopy(logits), copy.deepcopy(logits)
                plus[i][j] += h
                minus[i][j] -= h
                numerical = (objective_and_gradient(plus, trainer.reference, batch, beta=0.13)[0]
                             - objective_and_gradient(minus, trainer.reference, batch, beta=0.13)[0]) / (2*h)
                self.assertAlmostEqual(gradient[i][j], numerical, places=6)

    def test_learning_and_frozen_reference(self):
        trainer = BanditTrainer()
        reference = copy.deepcopy(trainer.reference)
        start = trainer.expected_reward()
        for _ in range(80):
            trainer.step()
        self.assertGreater(trainer.expected_reward(), 0.9)
        self.assertGreater(trainer.expected_reward(), start)
        self.assertEqual(trainer.reference, reference)
        self.assertEqual(trainer.actor_logits, trainer.rollout_logits)
        self.assertIsNot(trainer.actor_logits, trainer.rollout_logits)

    def test_checkpoint_restores_exact_next_updates(self):
        trainer = BanditTrainer()
        for _ in range(11):
            trainer.step()
        # Force the same conversion as a real checkpoint file.
        restored = BanditTrainer.from_state(json.loads(json.dumps(trainer.state_dict())))
        for _ in range(9):
            self.assertEqual(trainer.step(), restored.step())
        self.assertEqual(trainer.state_dict(), restored.state_dict())

    def test_unsynchronized_rollout_is_rejected(self):
        trainer = BanditTrainer()
        trainer.actor_version += 1
        with self.assertRaisesRegex(RuntimeError, "synchronized"):
            trainer.collect()


if __name__ == "__main__":
    unittest.main()
