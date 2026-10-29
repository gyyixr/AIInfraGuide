import math
import unittest
from mechanics_lab import online_attention, ring_allreduce, tensor_parallel_oracle


class MechanicsTests(unittest.TestCase):
    def test_ring_rounds_and_output(self):
        inputs = [[rank * 100 + chunk for chunk in range(4)] for rank in range(4)]
        r = ring_allreduce(inputs)
        self.assertEqual(r["expected_sum"], [600, 604, 608, 612])
        self.assertEqual(len(r["rounds"]), 6)
        self.assertEqual(r["reduce_scatter_ownership"][0], {"chunk": 1, "value": 604})
        self.assertEqual(inputs[0], [0, 1, 2, 3])

    def test_ring_other_group_sizes(self):
        for n in (2, 3, 5):
            ring_allreduce([[r - c for c in range(n)] for r in range(n)])

    def test_tp_forward_and_backward(self):
        r = tensor_parallel_oracle()
        self.assertEqual(r["output"], [[10, 14, 14, 20], [2, 10, 4, 14]])
        self.assertEqual(r["input_gradient"], [[3, 3, 7, 7], [7, 10, 15, 22]])

    def test_online_softmax_is_partition_invariant(self):
        scores, values = [1000, 1001, 998, 1002], [1, 2, 3, 4]
        outputs = [online_attention(scores, values, partition)["full_attention"]
                   for partition in ([4], [1, 3], [2, 2], [1, 1, 1, 1])]
        self.assertTrue(all(math.isclose(v, outputs[0], rel_tol=1e-14) for v in outputs))

    def test_averaging_local_attention_is_wrong(self):
        first = online_attention([0], [0], [1])["full_attention"]
        second = online_attention([math.log(3)], [4], [1])["full_attention"]
        merged = online_attention([0, math.log(3)], [0, 4], [1, 1])["full_attention"]
        self.assertAlmostEqual((first + second) / 2, 2)
        self.assertAlmostEqual(merged, 3)


if __name__ == "__main__":
    unittest.main()
