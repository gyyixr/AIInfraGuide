#!/usr/bin/env python3
"""Deterministic, standard-library numerical oracles; no real GPU collectives."""
import argparse
import copy
import json
import math
from pathlib import Path


def ring_allreduce(inputs):
    n = len(inputs)
    if n < 2 or any(len(row) != n for row in inputs):
        raise ValueError("this oracle uses N ranks and N scalar chunks per rank")
    buffers = copy.deepcopy(inputs)
    trace = []
    for step in range(n - 1):
        previous = copy.deepcopy(buffers)
        messages = []
        for sender in range(n):
            dest = (sender + 1) % n
            chunk = (sender - step) % n
            value = previous[sender][chunk]
            buffers[dest][chunk] += value
            messages.append({"src": sender, "dst": dest, "chunk": chunk, "value": value})
        trace.append({"phase": "reduce_scatter", "round": step + 1,
                      "messages": messages, "buffers": copy.deepcopy(buffers)})
    # This schedule leaves reduced chunk (rank+1)%N on rank, a cyclic ownership
    # permutation. It is not the rank-aligned output contract of an RS API.
    owned = {rank: {"chunk": (rank + 1) % n, "value": buffers[rank][(rank + 1) % n]}
             for rank in range(n)}
    for step in range(n - 1):
        previous = copy.deepcopy(buffers)
        messages = []
        for sender in range(n):
            dest = (sender + 1) % n
            chunk = (sender + 1 - step) % n
            value = previous[sender][chunk]
            buffers[dest][chunk] = value
            messages.append({"src": sender, "dst": dest, "chunk": chunk, "value": value})
        trace.append({"phase": "all_gather", "round": step + 1,
                      "messages": messages, "buffers": copy.deepcopy(buffers)})
    expected = [sum(row[j] for row in inputs) for j in range(n)]
    assert all(row == expected for row in buffers)
    return {"inputs": inputs, "reduce_scatter_ownership": owned,
            "rounds": trace, "expected_sum": expected, "outputs": buffers}


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def tensor_parallel_oracle():
    x = [[1, 2, 3, 4], [2, 1, 0, 3]]
    w = [[1, 0, 2, 0], [0, 1, 0, 2], [3, 0, 4, 0], [0, 3, 0, 4]]
    dy = [[1, 1, 1, 1], [1, 2, 3, 4]]
    expected = matmul(x, w)
    columns = [[[r[j] for j in group] for r in w] for group in ([0, 1], [2, 3])]
    y_column = [matmul(x, shard) for shard in columns]
    gathered = [left + right for left, right in zip(*y_column)]
    assert gathered == expected
    partial_y = [matmul([r[:2] for r in x], w[:2]),
                 matmul([r[2:] for r in x], w[2:])]
    assert add(*partial_y) == expected
    dx_parts = [matmul([r[:2] for r in dy], transpose(columns[0])),
                matmul([r[2:] for r in dy], transpose(columns[1]))]
    assert add(*dx_parts) == matmul(dy, transpose(w))
    dw_rows = matmul(transpose([r[:2] for r in x]), dy) + matmul(transpose([r[2:] for r in x]), dy)
    assert dw_rows == matmul(transpose(x), dy)
    return {"input": x, "weight": w, "output": expected,
            "column_shard_outputs": y_column, "row_shard_partial_outputs": partial_y,
            "column_parallel_input_gradient_parts": dx_parts,
            "input_gradient": add(*dx_parts), "weight_gradient": dw_rows}


def online_attention(scores, values, chunks):
    """One query, scalar values; merge unnormalized online-softmax statistics."""
    maximum, denominator, numerator = -math.inf, 0.0, 0.0
    trace = []
    start = 0
    for size in chunks:
        block_scores = scores[start:start + size]
        block_values = values[start:start + size]
        start += size
        local_max = max(block_scores)
        local_weights = [math.exp(s - local_max) for s in block_scores]
        local_den = sum(local_weights)
        local_num = sum(p * v for p, v in zip(local_weights, block_values))
        new_max = max(maximum, local_max)
        old_scale = math.exp(maximum - new_max)
        new_scale = math.exp(local_max - new_max)
        denominator = old_scale * denominator + new_scale * local_den
        numerator = old_scale * numerator + new_scale * local_num
        maximum = new_max
        trace.append({"max": maximum, "denominator": denominator,
                      "numerator": numerator, "output": numerator / denominator})
    if start != len(scores):
        raise ValueError("chunks must cover the whole sequence")
    full_max = max(scores)
    weights = [math.exp(s - full_max) for s in scores]
    expected = sum(p * v for p, v in zip(weights, values)) / sum(weights)
    assert math.isclose(trace[-1]["output"], expected, rel_tol=1e-14)
    return {"scores": scores, "values": values, "trace": trace, "full_attention": expected}


def run():
    ring = ring_allreduce([[100 * rank + chunk for chunk in range(4)] for rank in range(4)])
    tp = tensor_parallel_oracle()
    attention = online_attention([1000, 1001, 998, 1002], [1, 2, 3, 4], [2, 2])
    # Two local means receive equal rank weights, despite 1 vs 3 valid values.
    local_values = [[1.0], [3.0, 3.0, 3.0]]
    local_mean_average = sum(sum(v) / len(v) for v in local_values) / 2
    global_mean = sum(map(sum, local_values)) / sum(map(len, local_values))
    assert local_mean_average == 2 and global_mean == 2.5
    return {"scope": "deterministic mathematical simulation; not GPU or network execution",
            "ring": ring, "tensor_parallel": tp, "online_attention": attention,
            "unequal_counts": {"wrong_rank_mean": local_mean_average, "global_mean": global_mean}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ring_sum": result["ring"]["expected_sum"],
                      "tp_output": result["tensor_parallel"]["output"],
                      "attention": result["online_attention"]["full_attention"],
                      "unequal_counts": result["unequal_counts"]}, indent=2))
