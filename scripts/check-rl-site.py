#!/usr/bin/env python3
"""Validate built RL pages, local links/anchors, math, and content coverage.

Run after npm run build. External references are reviewed by the authors;
this checker deliberately checks local delivery without network dependencies.
"""
import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BASE = "/AIInfraGuide"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.math_errors = 0
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if "katex-error" in attrs.get("class", "").split():
            self.math_errors += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=("rl", "distributed"), default="rl")
    args = parser.parse_args()
    module = args.module
    source_directory = "模块五-LLM-RL后训练" if module == "rl" else "模块三-分布式训练"
    pages = sorted((DIST / module).rglob("*.html"))
    if not pages:
        raise SystemExit(f"No built {module} pages. Run npm run build first.")
    cache = {}
    errors = []
    checked_links = 0

    def read(path):
        if path not in cache:
            cache[path] = Page(path.read_text())
        return cache[path]

    docs = list((ROOT / "docs/guides" / source_directory).glob("*.md"))
    chapters, children = set(), {}
    for doc in docs:
        text = doc.read_text()
        title = re.search(r'^title:\s*"?(.*?)"?\s*$', text, re.M)
        chapter_title = re.match(r"第(\d+)章", title[1]) if title else None
        if chapter_title:
            chapters.add(int(chapter_title[1]))
        else:
            chapter = re.search(r"^chapter:\s*(\d+)\s*$", text, re.M)
            if not chapter:
                errors.append(f"{doc.name}: missing chapter metadata")
            else:
                children.setdefault(int(chapter[1]), []).append(doc.name)
    for chapter in chapters:
        if not children.get(chapter):
            errors.append(f"chapter {chapter}: no detailed articles")
    for chapter in children:
        if chapter not in chapters:
            errors.append(f"chapter {chapter}: no chapter introduction")
    if len(pages) != len(docs) + 1:
        errors.append(f"built {module} pages ({len(pages)}) do not match source docs + landing ({len(docs)+1})")

    for path in pages:
        page = read(path)
        if page.math_errors:
            errors.append(f"{path.relative_to(DIST)}: {page.math_errors} KaTeX errors")
        for href in page.links:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue
            if not parsed.path:
                target = path
            else:
                local = unquote(parsed.path)
                if local == BASE or local.startswith(BASE + "/"):
                    target = DIST / local[len(BASE):].lstrip("/")
                elif local.startswith("/"):
                    errors.append(f"{path.relative_to(DIST)}: missing base prefix: {href}")
                    continue
                else:
                    target = path.parent / local
                if target.is_dir():
                    target = target / "index.html"
            checked_links += 1
            if not target.is_file():
                errors.append(f"{path.relative_to(DIST)}: broken link: {href}")
            elif parsed.fragment and target.suffix == ".html":
                anchor = unquote(parsed.fragment)
                if anchor not in read(target).ids:
                    errors.append(f"{path.relative_to(DIST)}: missing anchor: {href}")

    result = {"module": module, "source_documents": len(docs), "chapters": len(chapters),
              "detailed_articles": sum(map(len, children.values())),
              "built_pages": len(pages), "local_links_checked": checked_links,
              "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
