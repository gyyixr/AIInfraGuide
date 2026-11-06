#!/usr/bin/env python3
"""Audit every built HTML page without making network requests.

Checks local navigation, anchors, image/script/style assets and KaTeX errors.
Run after npm run build; use --output to retain the JSON report.
External URLs are counted but deliberately not treated as verified.
"""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
BASE = "/AIInfraGuide/"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.references = []
        self.math_errors = 0
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        if "katex-error" in attrs.get("class", "").split():
            self.math_errors += 1
        if tag == "a" and "href" in attrs:
            self.references.append(("link", attrs["href"]))
        if tag in {"img", "script", "source", "video", "audio", "iframe"} and attrs.get("src"):
            self.references.append(("asset", attrs["src"]))
        if tag == "video" and attrs.get("poster"):
            self.references.append(("asset", attrs["poster"]))
        if tag == "link" and attrs.get("href") and set(attrs.get("rel", "").split()) & {
            "stylesheet", "icon", "preload", "modulepreload", "manifest"
        }:
            self.references.append(("asset", attrs["href"]))
        if tag in {"img", "source"} and attrs.get("srcset"):
            # Site image srcsets use ordinary URLs. Data URLs contain commas;
            # retain them as external references instead of splitting them.
            srcset = attrs["srcset"]
            candidates = [srcset] if srcset.lstrip().startswith("data:") else srcset.split(",")
            for candidate in candidates:
                parts = candidate.strip().split()
                if parts:
                    self.references.append(("asset", parts[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "dist")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    site = args.site.resolve()
    paths = sorted(site.rglob("*.html"))
    if not paths:
        raise SystemExit(f"No HTML pages in {site}; run npm run build first.")
    pages = {path: Page(path.read_text()) for path in paths}
    errors = []
    counters = {"local_links_checked": 0, "local_assets_checked": 0, "external_references_skipped": 0}
    routes = []
    for path, page in pages.items():
        relative = path.relative_to(site).as_posix()
        route = BASE + relative.removesuffix("index.html") if relative.endswith("index.html") else BASE + relative
        routes.append(route)
        if page.math_errors:
            errors.append({"page": relative, "error": "katex", "count": page.math_errors})
        for kind, reference in page.references:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc:
                counters["external_references_skipped"] += 1
                continue
            counters["local_links_checked" if kind == "link" else "local_assets_checked"] += 1
            resolved = urlsplit(urljoin(quote(route, safe="/"), reference))
            url_path = unquote(resolved.path)
            if url_path == BASE.rstrip("/"):
                url_path = BASE
            if not url_path.startswith(BASE):
                errors.append({"page": relative, "error": "missing_base_prefix", "url": reference})
                continue
            target = (site / url_path[len(BASE):]).resolve()
            if not target.is_relative_to(site):
                errors.append({"page": relative, "error": "outside_site", "url": reference})
                continue
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append({"page": relative, "error": "missing_target", "url": reference})
                continue
            if resolved.fragment and target.suffix == ".html":
                anchor = unquote(resolved.fragment).split(":~:text=", 1)[0]
                if anchor and anchor not in pages[target].ids:
                    errors.append({"page": relative, "error": "missing_anchor", "url": reference})
    report = {"built_pages": len(paths), **counters, "errors": errors, "routes": routes}
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(json.dumps({k: v for k, v in report.items() if k != "routes"}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
