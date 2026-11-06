#!/usr/bin/env python3
"""Verify synthetic MLA absorption + separate RoPE using ordinary PyTorch.

No DeepSeek weights, FlashMLA, DeepGEMM or serving engine are loaded.
"""
import argparse
import json
import math
from pathlib import Path

import torch


def rope(x, positions):
    dim = x.shape[-1]
    assert dim % 2 == 0
    inv_freq = 10000.0 ** (-torch.arange(0, dim, 2, dtype=x.dtype, device=x.device) / dim)
    shape = [1] * x.ndim
    shape[1] = len(positions)
    angles = torch.tensor(positions, dtype=x.dtype, device=x.device).reshape(shape)
    freq_shape = [1] * x.ndim
    freq_shape[-1] = dim // 2
    angles = angles * inv_freq.reshape(freq_shape)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * angles.cos() - odd * angles.sin(),
                        even * angles.sin() + odd * angles.cos()), dim=-1).reshape_as(x)


def attention(inputs, absorbed, include_rope=True):
    c, wk, wv, q, qr, kr = inputs
    seq_len, query_len = c.shape[1], q.shape[1]
    query_positions = list(range(seq_len - query_len, seq_len))
    qr = rope(qr, query_positions)
    kr = rope(kr, list(range(seq_len)))
    if absorbed:
        q_abs = torch.einsum('bthd,hrd->bthr', q, wk)
        scores = torch.einsum('bthr,bsr->bhts', q_abs, c)
    else:
        keys = torch.einsum('bsr,hrd->bshd', c, wk)
        scores = torch.einsum('bthd,bshd->bhts', q, keys)
    if include_rope:
        scores = scores + torch.einsum('bthd,bsd->bhts', qr, kr)
    scores = scores / math.sqrt(q.shape[-1] + qr.shape[-1])
    valid = torch.arange(seq_len, device=c.device)[None, :] <= torch.tensor(query_positions, device=c.device)[:, None]
    logits = scores.masked_fill(~valid[None, None], -torch.inf)
    probabilities = logits.softmax(-1)
    if absorbed:
        latent = torch.einsum('bhts,bsr->bthr', probabilities, c)
        output = torch.einsum('bthr,hrv->bthv', latent, wv)
    else:
        values = torch.einsum('bsr,hrv->bshv', c, wv)
        output = torch.einsum('bhts,bshv->bthv', probabilities, values)
    return scores, output


def run(device):
    torch.set_num_threads(1)
    generator = torch.Generator(device='cpu').manual_seed(7)
    shapes = [(2,8,4), (2,4,6), (2,4,5), (2,3,2,6), (2,3,2,4), (2,8,4)]
    base = [torch.randn(shape, generator=generator, dtype=torch.float64) * 0.4 for shape in shapes]
    left = [x.to(device).detach().requires_grad_() for x in base]
    right = [x.to(device).detach().clone().requires_grad_() for x in base]
    full_logits, full_output = attention(left, absorbed=False)
    abs_logits, abs_output = attention(right, absorbed=True)
    full_grad = torch.autograd.grad(full_output.square().mean(), left)
    abs_grad = torch.autograd.grad(abs_output.square().mean(), right)
    logit_error = (full_logits - abs_logits).abs().max().item()
    output_error = (full_output - abs_output).abs().max().item()
    grad_error = max((a-b).abs().max().item() for a,b in zip(full_grad, abs_grad))
    _, wrong_output = attention([x.detach() for x in left], absorbed=True, include_rope=False)
    missing_rope_error = (wrong_output - full_output).abs().max().item()
    assert logit_error < 1e-12 and output_error < 1e-12 and grad_error < 1e-12
    assert missing_rope_error > 1e-7
    if device == 'cuda':
        torch.cuda.synchronize()
    return {
        'scope': 'synthetic MLA algebra + separate RoPE; ordinary PyTorch forward/backward; not official model/kernel execution',
        'torch': torch.__version__, 'device': device,
        'hardware': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
        'capability': list(torch.cuda.get_device_capability(0)) if device == 'cuda' else None,
        'dtype': 'float64', 'input_shapes': shapes, 'seed': 7,
        'logit_max_abs_error': logit_error,
        'output_max_abs_error': output_error,
        'gradient_max_abs_error': grad_error,
        'omitted_rope_negative_control_error': missing_rope_error,
        'checks': 'passed',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable in this process; CPU run does not require CUDA.')
    result = run(args.device)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
