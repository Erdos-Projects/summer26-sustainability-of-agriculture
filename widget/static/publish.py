"""Publish the built static site to the orphan `gh-pages` branch GitHub Pages serves.

    python -m widget.static.export            # build            -> ./static-site/
    python -m widget.static.publish           # stage + describe (does NOT push)
    python -m widget.static.publish --push    # force-push       -> origin/gh-pages

WHY A BRANCH AND NOT A DIRECTORY. GitHub Pages' "Deploy from a branch" only offers `/` or `/docs`
as the source folder, so it cannot be pointed at ./static-site/. Publishing the directory as the
ROOT of a separate branch gets the layout we want without renaming anything, and keeps main clean:
static-site/ is gitignored, so none of the ~90 MB of generated binaries ever lands in main's
history. Point Pages at: Deploy from a branch -> gh-pages -> / (root).

WHY NO HISTORY. Each publish writes a commit with NO PARENT and force-pushes it, so gh-pages is
always exactly one commit holding exactly the current site. Rebuilds replace it rather than
accumulating, which matters when every rebuild is tens of megabytes of binary artifacts.

HOW IT AVOIDS TOUCHING YOUR WORKING TREE. It never checks anything out, creates no local branch,
and does not run `git add` against your real index. It builds the tree with plumbing -- a
throwaway index file, `write-tree`, `commit-tree` -- against GIT_WORK_TREE=static-site/, then
pushes the resulting commit object straight to the remote ref. Your branch, index and working tree
are untouched whether or not the push happens.

The push is FORCE and it is REMOTE, so it never happens without --push.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # widget/static
_ROOT = _HERE.parent.parent

DEFAULT_SITE = _ROOT / "static-site"
BRANCH = "gh-pages"
REMOTE = "origin"


def _git(*args, env=None, capture=True) -> str:
    """Run git at the repo root, raising with the stderr text on failure."""
    r = subprocess.run(
        ["git", "-C", str(_ROOT), *args],
        env={**os.environ, **(env or {})},
        capture_output=capture,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{(r.stderr or '').strip()}")
    return (r.stdout or "").strip()


def _check_site(site: Path) -> tuple[int, int]:
    """Fail unless `site` looks like a finished export. Returns (n_files, total_bytes)."""
    if not site.is_dir():
        raise SystemExit(f"{site} does not exist. Run `python -m widget.static.export` first.")
    missing = [n for n in ("index.html", ".nojekyll") if not (site / n).exists()]
    if missing:
        raise SystemExit(
            f"{site} is missing {missing} -- that is not a finished export. "
            "Re-run `python -m widget.static.export`."
        )
    files = [f for f in site.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def publish(site: Path = DEFAULT_SITE, push: bool = False, branch: str = BRANCH, remote: str = REMOTE) -> str:
    n_files, total = _check_site(site)
    print(f"{site.relative_to(_ROOT)}: {n_files} files, {total / 1e6:.1f} MB")
    if total > 1e9:
        print("  WARNING: over GitHub Pages' 1 GB published-site limit")

    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index"), "GIT_WORK_TREE": str(site)}
        # --force because static-site/ is gitignored; without it every path would be skipped and
        # the commit would be empty.
        _git("add", "--all", "--force", ".", env=env)
        tree = _git("write-tree", env=env)

    head = _git("rev-parse", "--short", "HEAD")
    desc = _git("log", "-1", "--format=%s", "HEAD")
    # No parent: an orphan commit, so the branch never accumulates history.
    commit = _git("commit-tree", tree, "-m", f"Publish static site from {head} ({desc})")

    print(f"staged commit {commit[:12]} (tree {tree[:12]}, orphan) from main @ {head}")
    if not push:
        print(
            f"\nnothing pushed. To publish:\n"
            f"    python -m widget.static.publish --push\n"
            f"which force-updates {remote}/{branch} to this commit."
        )
        return commit

    print(f"force-pushing -> {remote}/{branch} ...")
    _git("push", "--force", remote, f"{commit}:refs/heads/{branch}", capture=False)
    url = _git("remote", "get-url", remote)
    slug = url.rstrip("/").removesuffix(".git").split(":")[-1].split("/")[-2:]
    print(
        f"published.\n"
        f"  Set Pages to: Deploy from a branch -> {branch} -> / (root)\n"
        f"  https://{slug[0].lower()}.github.io/{slug[1]}/"
    )
    return commit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", type=Path, default=DEFAULT_SITE, help=f"built site directory (default {DEFAULT_SITE})")
    p.add_argument("--push", action="store_true", help="actually force-push; without it, stage and describe only")
    p.add_argument("--branch", default=BRANCH)
    p.add_argument("--remote", default=REMOTE)
    a = p.parse_args()
    sys.exit(0 if publish(site=a.site, push=a.push, branch=a.branch, remote=a.remote) else 1)
