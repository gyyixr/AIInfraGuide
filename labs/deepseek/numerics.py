#!/usr/bin/env python3
"""Small, dependency-free math/state oracles; no official accelerator kernels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


def block_scaled_gemm(a, b, sa, sb, block_k):
    m, k, n = len(a), len(a[0]), len(b)
    if k % block_k or any(len(row) != k for row in a + b):
        raise ValueError("invalid toy GEMM shape")
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for g in range(k // block_k):
                partial = sum(a[i][t] * b[j][t]
                              for t in range(g * block_k, (g + 1) * block_k))
                out[i][j] += partial * sa[i][g] * sb[g]
    return out


def expanded_ranges(psum, alignment):
    start, result = 0, []
    for end in psum:
        if end < start:
            raise ValueError("invalid expanded count")
        result.append((start, end))
        start = ((end + alignment - 1) // alignment) * alignment
    return result


def moe_example():
    inputs = [2.0, 3.0]
    routes = [[(0, 0.25), (2, 0.75)], [(1, 0.6), (2, 0.4)]]
    coefficients = [1.0, 2.0, 3.0]
    rows, output, dx = [], [0.0, 0.0], [0.0, 0.0]
    for token, selected in enumerate(routes):
        for expert, gate in selected:
            value = coefficients[expert] * inputs[token]
            rows.append({"token": token, "expert": expert, "gate": gate,
                         "expert_output": value, "weighted_output": gate * value})
            output[token] += gate * value
            dx[token] += gate * coefficients[expert]
    return {"expanded_rows": rows, "output": output, "input_gradient": dx}


def scan_filter_topk(values, k, block_size, extra, order):
    if not 1 <= k <= len(values) or not all(math.isfinite(x) for x in values):
        raise ValueError("finite input and a valid k are required")
    indexed = list(enumerate(values))
    blocks = [indexed[i:i + block_size] for i in range(0, len(values), block_size)]
    if sorted(order) != list(range(len(blocks))):
        raise ValueError("order must be a block permutation")
    candidate, threshold, work = [], -math.inf, 0
    for block_id in order:
        candidate.extend((value, index) for index, value in blocks[block_id] if value > threshold)
        if len(candidate) >= k + extra:
            work += len(candidate)
            candidate = sorted(candidate, reverse=True)[:k]
            threshold = candidate[-1][0]
    work += len(candidate)
    return sorted(candidate, reverse=True)[:k], work


def sinkhorn(matrix, iterations=100):
    p = [list(map(float, row)) for row in matrix]
    if not p or any(len(row) != len(p) or min(row) <= 0 for row in p):
        raise ValueError("positive square matrix required")
    for _ in range(iterations):
        p = [[x / sum(row) for x in row] for row in p]
        sums = [sum(row[j] for row in p) for j in range(len(p))]
        p = [[x / sums[j] for j, x in enumerate(row)] for row in p]
    return p


def physical_indices(logical_ids, block_table, page_size):
    return [block_table[i // page_size] * page_size + i % page_size for i in logical_ids]


def decode_fp4_byte(packed, scale):
    levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    def decode(code):
        return (-1.0 if code & 8 else 1.0) * levels[code & 7] * scale
    return [decode(packed & 15), decode(packed >> 4)]


def attention_with_sink(scores, values, sink_logit=None):
    if not scores:
        # Mathematical empty-set result, not FlashMLA's +inf sentinel contract.
        return {"output": 0.0, "lse_without_sink": -math.inf}
    maximum = max(scores + ([] if sink_logit is None else [sink_logit]))
    weights = [math.exp(s - maximum) for s in scores]
    denominator = sum(weights)
    lse = maximum + math.log(denominator)
    sink_weight = 0.0 if sink_logit is None else math.exp(sink_logit - maximum)
    output = sum(w * v for w, v in zip(weights, values)) / (denominator + sink_weight)
    return {"output": output, "lse_without_sink": lse}


def speculative_distribution(p, q):
    if len(p) != len(q) or any(x < 0 for x in p + q):
        raise ValueError("invalid probability vectors")
    if not math.isclose(sum(p), 1) or not math.isclose(sum(q), 1):
        raise ValueError("probabilities must sum to one")
    accepted = [min(a, b) for a, b in zip(p, q)]
    residual_mass = [max(a - b, 0.0) for a, b in zip(p, q)]
    return {"accepted_mass": accepted, "rejection_probability": sum(residual_mass),
            "output_distribution": [a + r for a, r in zip(accepted, residual_mass)]}


def greedy_prefix_plan(confidences, steps_per_second):
    """Toy causal-frontier planner with fixed, predeclared probabilities.

    Real generation additionally requires the information boundary discussed
    in chapter 5.3. This function does not implement production ZOS history.
    """
    requests = len(confidences)
    admitted, survival = [0] * requests, [1.0] * requests
    expected = float(requests)
    throughput = expected * steps_per_second[requests]
    trace = [{"lengths": admitted[:], "expected": expected, "throughput": throughput}]
    while True:
        frontier = [(survival[i] * c[admitted[i]], i) for i, c in enumerate(confidences)
                    if admitted[i] < len(c)]
        if not frontier:
            break
        benefit, chosen = max(frontier)
        new_total = requests + sum(admitted) + 1
        candidate = (expected + benefit) * steps_per_second[new_total]
        if candidate < throughput:
            break
        admitted[chosen] += 1
        survival[chosen] = benefit
        expected += benefit
        throughput = candidate
        trace.append({"lengths": admitted[:], "expected": expected, "throughput": throughput})
    return {"lengths": admitted, "trace": trace}


def teaching_cache_key(source, flags, compiler, dependencies, extra):
    record = {"source": source, "flags": list(flags), "compiler": compiler,
              "dependencies": dependencies, "extra": extra}
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Replica:
    committed: tuple[int, str] = (0, "old")
    pending: tuple[int, str] | None = None

    def strict_read(self):
        return {"status": "retry"} if self.pending else {"status": "ok", "value": self.committed}

    def commit(self):
        if self.pending is None:
            raise ValueError("nothing pending")
        self.committed, self.pending = self.pending, None


def chain_visibility_example():
    replicas = [Replica() for _ in range(3)]
    for replica in replicas:
        replica.pending = (1, "new")
    replicas[2].commit()
    tail_committed = [r.strict_read() for r in replicas]
    replicas[1].commit()
    replicas[0].commit()
    return {"tail_committed": tail_committed, "all_acked": [r.strict_read() for r in replicas],
            "scope": "single-process state illustration; not a storage protocol proof"}


def run():
    topk, work = scan_filter_topk([3, 1, 8, 2, 7, 5, 6, 4, 9, 0, 12, 10], 3, 4, 3, [1, 0, 2])
    matrix = sinkhorn([[1, 2, 3], [4, 2, 1], [2, 1, 5]])
    return {
        "scope": "CPU mathematical and state oracles; not official DeepSeek accelerator execution",
        "block_scaled_gemm": block_scaled_gemm([[1,2,3,4],[2,0,1,3]], [[2,1,0,1],[1,3,2,0]],
                                               [[0.5,1],[2,0.25]], [1,2], 2),
        "moe": moe_example(), "expanded_ranges": expanded_ranges([3,9,12,14],4),
        "topk": {"values_and_indices": topk, "compact_work": work},
        "sinkhorn": {"matrix": matrix, "row_sums": list(map(sum, matrix)),
                     "column_sums": [sum(row[j] for row in matrix) for j in range(3)]},
        "physical_indices": physical_indices([1,4,5],[2,5],4),
        "fp4": decode_fp4_byte(0x62,0.5),
        "attention_sink": attention_with_sink([0,math.log(3)],[2,6],math.log(2)),
        "speculative_sampling": speculative_distribution([0.6,0.4],[0.5,0.5]),
        "selection_bias_counterexample": {"target":[0.6,0.4],"wrong_output":[0.8,0.2]},
        "prefix_plan": greedy_prefix_plan([[0.9,0.6,0.4],[0.7,0.7,0.2]],
                                          {2:100,3:90,4:80,5:60,6:55,7:48,8:40}),
        "cache_key": teaching_cache_key("kernel",["-DA=1","-DA=2"],"compiler",{},"abi-1"),
        "chain_visibility": chain_visibility_example(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
