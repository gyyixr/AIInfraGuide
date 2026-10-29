#!/usr/bin/env python3
"""Two-process CPU/Gloo correctness lab; no CUDA or model downloads.

Run: python -m torch.distributed.run --standalone --nproc-per-node=2 \
     labs/distributed/ddp_lab.py --output labs/distributed/results/latest-ddp.json
"""
from __future__ import annotations

import argparse
import copy
import json
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def dataset(rank):
    # Two microbatches per rank, with deliberately unequal valid counts: 3 vs 5.
    x = (torch.arange(12, dtype=torch.float64).reshape(6, 2) + rank * 3) / 10
    y = x @ torch.tensor([[0.7], [-0.4]], dtype=torch.float64) + 0.2
    mask = torch.tensor([1, 1, 0, 1, 0, 0] if rank == 0 else [1, 1, 1, 1, 1, 0], dtype=torch.bool)
    return x, y, mask


def model():
    m = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[0.1, -0.2]], dtype=torch.float64))
        m.bias.fill_(0.05)
    return m


def max_difference(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def optimizer_state_error(left, right):
    errors = []
    for lp, rp in zip(left.param_groups[0]["params"], right.param_groups[0]["params"]):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            errors.append((left.state[lp][key] - right.state[rp][key]).abs().max().item())
    return max(errors)


def check_collectives(rank, world):
    total = torch.tensor([float(rank + 1)], dtype=torch.float64)
    work = dist.all_reduce(total, async_op=True)
    work.wait()  # CPU/Gloo completion; not a CUDA stream timing example.
    assert total.item() == world * (world + 1) / 2
    source = torch.tensor([rank], dtype=torch.int64)
    gathered = [torch.empty_like(source) for _ in range(world)]
    dist.all_gather(gathered, source)
    assert [v.item() for v in gathered] == list(range(world))
    inputs = torch.arange(world, dtype=torch.int64) + 10 * rank
    shard = torch.empty(1, dtype=torch.int64)
    dist.reduce_scatter_tensor(shard, inputs)
    assert shard.item() == 10 * world * (world - 1) // 2 + world * rank
    received = torch.empty_like(inputs)
    dist.all_to_all_single(received, inputs)
    assert received.tolist() == [10 * src + rank for src in range(world)]
    return {"rank": rank, "all_reduce": total.item(),
            "all_gather": [v.item() for v in gathered],
            "reduce_scatter": shard.item(), "all_to_all": received.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2 to exercise Adam state")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if world != 2:
            raise ValueError("This teaching dataset requires exactly 2 ranks")
        collective_result = check_collectives(rank, world)
        actor = DDP(model(), broadcast_buffers=False)
        oracle = model()
        optimizer = torch.optim.AdamW(actor.parameters(), lr=0.01, weight_decay=0.02)
        reference_optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.01, weight_decay=0.02)
        x, y, mask = dataset(rank)
        all_x = torch.cat([dataset(r)[0] for r in range(world)])
        all_y = torch.cat([dataset(r)[1] for r in range(world)])
        all_mask = torch.cat([dataset(r)[2] for r in range(world)])
        global_count = mask.sum().clone()
        dist.all_reduce(global_count)
        assert global_count.item() == 8

        # A deliberate counterexample: averaging rank-local means is biased
        # relative to the global valid-element mean when counts differ.
        bad = model()
        bad_loss = ((bad(x) - y).square().flatten()[mask]).mean()
        bad_loss.backward()
        for p in bad.parameters():
            dist.all_reduce(p.grad)
            p.grad.div_(world)
        correct = model()
        ((correct(all_x) - all_y).square().flatten()[all_mask]).mean().backward()
        bad_gradient_error = max_difference([p.grad for p in bad.parameters()],
                                            [p.grad for p in correct.parameters()])
        assert bad_gradient_error > 1e-3

        # Resume oracle is loaded after step 1; all tensors stay on CPU.
        resumed = None
        resumed_optimizer = None
        history = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            reference_loss = ((oracle(all_x) - all_y).square().flatten()[all_mask]).mean()
            reference_loss.backward()
            local_sum = torch.tensor(0.0, dtype=torch.float64)
            for micro in range(2):
                sl = slice(micro * 3, (micro + 1) * 3)
                # no_sync must enclose BOTH forward and backward.
                context = actor.no_sync() if micro == 0 else nullcontext()
                with context:
                    losses = (actor(x[sl]) - y[sl]).square().flatten()
                    loss_sum = losses[mask[sl]].sum()
                    # DDP averages gradients across world; compensate that
                    # average so the result is sum(all valid losses)/N_global.
                    scaled = loss_sum * world / global_count
                    scaled.backward()
                    local_sum += loss_sum.detach()
            gradient_error = max_difference([p.grad for p in actor.parameters()],
                                            [p.grad for p in oracle.parameters()])
            assert gradient_error < 1e-12, gradient_error
            # One global norm clip AFTER accumulation and synchronized gradients.
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.2)
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 0.2)
            optimizer.step()
            reference_optimizer.step()
            param_error = max_difference(actor.parameters(), oracle.parameters())
            moments_error = optimizer_state_error(optimizer, reference_optimizer)
            assert param_error < 1e-12 and moments_error < 1e-12
            dist.all_reduce(local_sum)
            global_loss = (local_sum / global_count).item()
            assert abs(global_loss - reference_loss.item()) < 1e-12
            resume_error = None
            if resumed is not None:
                resumed_optimizer.zero_grad(set_to_none=True)
                resume_loss = ((resumed(all_x) - all_y).square().flatten()[all_mask]).mean()
                resume_loss.backward()
                torch.nn.utils.clip_grad_norm_(resumed.parameters(), 0.2)
                resumed_optimizer.step()
                resume_error = max_difference(resumed.parameters(), oracle.parameters())
                assert resume_error < 1e-12
                assert optimizer_state_error(resumed_optimizer, reference_optimizer) < 1e-12
            if step == 0:
                checkpoint = copy.deepcopy({"model": oracle.state_dict(),
                                            "optimizer": reference_optimizer.state_dict()})
                resumed = model()
                resumed.load_state_dict(checkpoint["model"])
                resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01, weight_decay=0.02)
                resumed_optimizer.load_state_dict(checkpoint["optimizer"])
            history.append({"step": step + 1, "global_mean_loss": global_loss,
                            "gradient_max_error": gradient_error,
                            "parameter_max_error": param_error,
                            "optimizer_state_max_error": moments_error,
                            "restored_oracle_parameter_error": resume_error})

        flat = torch.cat([p.detach().flatten() for p in actor.parameters()])
        all_params = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(all_params, flat)
        assert max_difference(all_params, [flat] * world) == 0
        rank_reports = [None] * world
        dist.all_gather_object(rank_reports, collective_result)
        if rank == 0:
            result = {
                "scope": "two actual CPU/Gloo processes; float64 linear model; no GPU/NCCL/FSDP performance claim",
                "torch": torch.__version__, "world_size": world, "backend": "gloo",
                "valid_elements_per_rank": [3, 5], "global_valid_elements": 8,
                "wrong_local_mean_gradient_error": bad_gradient_error,
                "collectives": rank_reports, "history": history,
                "final_parameters": flat.tolist(), "checks": "passed",
                "resume_scope": "in-memory deterministic oracle model+AdamW restore; not distributed durable checkpoint",
            }
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
