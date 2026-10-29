import { rm } from 'node:fs/promises';

// Pagefind is generated after Astro builds. Keeping the previous public copy
// would copy stale indexes into dist and make `cp -r` nest pagefind/pagefind
// on subsequent builds. Only clear this generated, gitignored directory.
await rm(new URL('../public/pagefind/', import.meta.url), {
  recursive: true,
  force: true,
});
