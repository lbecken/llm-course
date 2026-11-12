"""
Documentation integrity checks (run by `make check-docs` and CI).

1. Relative Markdown links point to files that exist (and anchors to headings that exist).
2. Repository paths written in backticks (`labs/...py`, `mlle/...`, `scripts/...`, ...) exist.
3. Links to nanochat source at the pinned commit point to existing files, and line ranges
   lie inside the file (needs the checkout; skipped with --no-upstream).
4. No link points to a nanochat commit other than the pin.
5. Special tokens inside Markdown tables use escaped pipes.
6. No leftover placeholders (TODO / TBD / FIXME) in course material.

Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mlle.nanochat_bridge import PINNED_SHA, nanochat_dir  # noqa: E402

SKIP_DIRS = {"external", ".git", ".venv", "venv", "node_modules", "results/runs"}
LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")
REF_DEF_RE = re.compile(r"^\[[^\]]+\]:\s*(\S+)", re.M)
NANOCHAT_TESTS = {"tests/test_engine.py", "tests/test_attention_fallback.py", "tests/test_optim.py",
                  "tests/test_tokenizer.py", "tests/test_execution.py", "tests/test_tasks.py"}
UPSTREAM_RE = re.compile(r"https://github\.com/karpathy/nanochat/(blob|tree)/([0-9a-f]{7,40})/([^\s)#>\]]*)(?:#L(\d+)(?:-L(\d+))?)?")
PATH_RE = re.compile(r"`((?:labs|mlle|scripts|tests|projects|docs|modules|solutions|results|resources|instructor|diagrams)/[A-Za-z0-9_./\-]+)`")
PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)


def md_files():
    for p in sorted(ROOT.rglob("*.md")):
        rel = p.relative_to(ROOT).as_posix()
        if any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS):
            continue
        yield p


def slugify(heading: str) -> str:
    """GitHub's anchor algorithm, closely enough: lowercase, drop punctuation, spaces -> hyphens."""
    h = re.sub(r"`", "", heading.strip().lower())
    h = re.sub(r"[^\w\- ]", "", h, flags=re.UNICODE)
    return h.replace(" ", "-")


def anchors_of(path: Path) -> set[str]:
    text = CODE_FENCE_RE.sub("", path.read_text(encoding="utf-8"))
    out, seen = set(), {}
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*$", line)
        if m:
            slug = slugify(m.group(2))
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            out.add(slug if n == 0 else f"{slug}-{n}")
    out |= set(re.findall(r'<a (?:id|name)="([^"]+)"', text))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-upstream", action="store_true", help="do not require the nanochat checkout")
    args = ap.parse_args()

    errors: list[str] = []
    upstream = nanochat_dir()
    have_upstream = (upstream / "nanochat" / "gpt.py").exists() and not args.no_upstream
    n_links = n_paths = n_upstream = 0

    for md in md_files():
        rel = md.relative_to(ROOT).as_posix()
        raw = md.read_text(encoding="utf-8")
        prose = CODE_FENCE_RE.sub("", raw)

        # 1. relative links
        for target in LINK_RE.findall(prose) + REF_DEF_RE.findall(prose):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            n_links += 1
            file_part, _, anchor = target.partition("#")
            dest = md if file_part == "" else (md.parent / file_part).resolve()
            if not dest.exists():
                errors.append(f"{rel}: broken link -> {target}")
                continue
            if anchor and dest.suffix == ".md" and anchor not in anchors_of(dest):
                errors.append(f"{rel}: missing anchor #{anchor} in {dest.relative_to(ROOT)}")

        # 2. backticked repository paths
        for path in PATH_RE.findall(prose):
            n_paths += 1
            candidate = path.rstrip("/.")
            if "*" in candidate or "NN" in candidate or "<" in candidate:
                continue
            if candidate.startswith("results/runs"):
                continue  # the learner's own outputs (git-ignored)
            if re.fullmatch(r"scripts/[a-z_]+\.py", candidate):
                continue  # nanochat's scripts live directly in scripts/; course scripts are in sub-folders
            if candidate in NANOCHAT_TESTS and (not have_upstream or (upstream / candidate).exists()):
                continue  # nanochat's own test files, referred to by their path inside external/nanochat
            if not (ROOT / candidate).exists():
                errors.append(f"{rel}: path does not exist -> {path}")

        # 3 + 4. upstream source links
        for kind, sha, path, l1, l2 in UPSTREAM_RE.findall(raw):
            n_upstream += 1
            if not PINNED_SHA.startswith(sha):
                errors.append(f"{rel}: link to nanochat commit {sha}, not the pin {PINNED_SHA[:7]}")
                continue
            if not have_upstream:
                continue
            target = upstream / path
            if not target.exists():
                errors.append(f"{rel}: upstream path missing at pin -> {path}")
                continue
            if l1 and target.is_file():
                n_lines = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
                last = int(l2 or l1)
                if int(l1) > last or last > n_lines:
                    errors.append(f"{rel}: {path}#L{l1}-L{last} outside file (has {n_lines} lines)")

        # 5. unescaped special tokens in table rows
        for i, line in enumerate(prose.splitlines(), 1):
            if line.lstrip().startswith("|") and re.search(r"<\|[a-z_]+\|>", line):
                errors.append(f"{rel}:{i}: special token in a table needs escaped pipes (<\\|name\\|>)")

        # 6. placeholders (allowed in this script's own documentation and CONTRIBUTING's rule text)
        if rel not in ("CONTRIBUTING.md", "PROGRESS.md"):
            for i, line in enumerate(prose.splitlines(), 1):
                if PLACEHOLDER_RE.search(line) and "nanochat" not in line.lower():
                    errors.append(f"{rel}:{i}: placeholder text -> {line.strip()[:80]}")

    print(f"checked {n_links} relative links, {n_paths} repository paths, {n_upstream} upstream links "
          f"({'verified against checkout' if have_upstream else 'pin only; checkout not used'})")
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    print("all documentation checks passed")


if __name__ == "__main__":
    main()
