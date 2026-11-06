import { spawnSync } from 'node:child_process';

for (const module of ['prerequisites', 'cuda', 'distributed', 'inference', 'rl', 'performance', 'deepseek']) {
  const result = spawnSync('python3', ['scripts/check-rl-site.py', '--module', module], { stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}

const site = spawnSync('python3', ['scripts/check-built-site.py'], { stdio: 'inherit' });
if (site.error) throw site.error;
if (site.status !== 0) process.exit(site.status || 1);
