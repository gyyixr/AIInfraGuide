#!/usr/bin/env python3
"""Run the pinned, publicly available EPLB planning algorithm on CPU.

Supply eplb.py from deepseek-ai/EPLB at the commit below. This script verifies
the normalized source digest before loading it; it never downloads code.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

COMMIT = 'd52c72d5b2f2fb4c41afbf8eb21366820239913d'
SOURCE_SHA256 = '25b37ccd83aef2369c68f2b4bb596608a4f289956d97d9a3504bfd3159459e29'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.source.read_text().replace('\r\n', '\n').rstrip('\n') + '\n'
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f'Source digest differs from pinned {COMMIT}: {digest}')
    spec = importlib.util.spec_from_file_location('pinned_eplb', args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loads = torch.tensor([[120, 60, 30, 30]])
    phy, log, counts = module.rebalance_experts(loads, 8, 1, 1, 4)
    replica_loads = (loads.float() / counts).gather(1, phy)
    gpu_loads = replica_loads.reshape(1, 4, 2).sum(-1)
    assert counts.tolist() == [[4, 2, 1, 1]]
    assert gpu_loads.tolist() == [[60, 60, 60, 60]]
    for expert in range(4):
        slots = log[0, expert, :counts[0, expert]].tolist()
        assert all(phy[0, slot].item() == expert for slot in slots)
    result = {
        'source': 'deepseek-ai/EPLB', 'commit': COMMIT,
        'normalized_source_sha256': digest, 'torch': torch.__version__, 'device': 'cpu',
        'logical_loads': loads.tolist(), 'physical_to_logical': phy.tolist(),
        'logical_to_physical': log.tolist(), 'replica_counts': counts.tolist(),
        'estimated_gpu_loads': gpu_loads.tolist(),
        'scope': 'Executed pinned planning code on CPU; four modeled GPUs, no GPU expert execution or migration',
        'checks': 'passed',
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
