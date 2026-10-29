import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// Set AIINFRA_PYTHON to an interpreter with PyTorch/Gloo installed.
// This runner never installs dependencies or accesses a GPU.
const python = process.env.AIINFRA_PYTHON || 'python3';
const child = spawn(python, [
  '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
  'labs/distributed/ddp_lab.py',
  '--output', 'labs/distributed/results/latest-ddp.json',
], {
  cwd: fileURLToPath(new URL('../', import.meta.url)),
  stdio: 'inherit',
  env: { ...process.env, OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || '1' },
});
child.on('error', (error) => {
  console.error(`Cannot start ${python}: ${error.message}. Set AIINFRA_PYTHON to your PyTorch interpreter.`);
  process.exitCode = 1;
});
child.on('exit', (code) => { process.exitCode = code ?? 1; });
