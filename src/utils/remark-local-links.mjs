import { readFileSync, statSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { slug } from 'github-slugger';

// Keep source-relative Markdown links usable on GitHub and map them to the
// same IDs as Astro's glob loader when building the local/static website.
const projectRoot = fileURLToPath(new URL('../../', import.meta.url));
const guideRoutes = {
  prerequisites: '/prerequisites',
  'cuda-optimization': '/cuda',
  'distributed-training': '/distributed',
  'inference-optimization': '/inference',
  'llm-rl': '/rl',
  'performance-analysis': '/performance',
  'deepseek-open': '/deepseek',
};
const collections = [
  { root: resolve(projectRoot, 'docs/guides'), route: null },
  { root: resolve(projectRoot, 'docs/posts'), route: '/blog' },
  { root: resolve(projectRoot, 'docs/interview'), route: '/interview' },
];

export default function remarkLocalLinks({ base = '/AIInfraGuide' } = {}) {
  const cache = new Map();
  const basePath = base === '/' ? '' : base.replace(/\/$/, '');

  function publishedPath(target) {
    const collection = collections.find(({ root }) => target.startsWith(root + sep));
    if (!collection) return null;
    const info = statSync(target);
    const key = `${info.mtimeMs}:${info.size}`;
    if (cache.get(target)?.key === key) return cache.get(target).url;
    const source = readFileSync(target, 'utf8');
    const frontmatter = source.match(/^---\s*\r?\n([\s\S]*?)\r?\n---/)?.[1] || '';
    if (/^draft:\s*true\s*$/m.test(frontmatter)) {
      throw new Error(`Linked document is a draft: ${relative(projectRoot, target)}`);
    }
    const category = frontmatter.match(/^category:\s*["']?([\w-]+)["']?\s*$/m)?.[1];
    const route = collection.route || guideRoutes[category] || '/guides';
    const customSlug = frontmatter.match(/^slug:\s*["']?([^\r\n"']+)["']?\s*$/m)?.[1];
    const generatedSlug = relative(collection.root, target).replace(/\.md$/i, '')
      .split(sep).map((segment) => slug(segment)).join('/').replace(/\/index$/, '');
    const url = `${basePath}${route}/${customSlug || generatedSlug}/`;
    cache.set(target, { key, url });
    return url;
  }

  return (tree, file) => {
    if (!file.path) return;
    const walk = (node) => {
      if ((node.type === 'link' || node.type === 'definition') && typeof node.url === 'string') {
        // External URLs, images and non-Markdown resources retain their meaning.
        if (!/^[a-z][a-z0-9+.-]*:|^\/\//i.test(node.url)) {
          const match = node.url.match(/^([^?#]+\.md)([?#].*)?$/i);
          if (match && !match[1].startsWith('/')) {
            try {
              const target = resolve(dirname(file.path), decodeURIComponent(match[1]));
              const url = publishedPath(target);
              if (url) node.url = url + (match[2] || '');
            } catch (error) {
              file.fail(`Cannot resolve local document link ${node.url}: ${error.message}`, node);
            }
          }
        }
      }
      for (const child of node.children || []) walk(child);
    };
    walk(tree);
  };
}
