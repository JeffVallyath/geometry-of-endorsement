"""Hash every file in the results package and refuse to finalize a leaky one.

--stage frozen   run after the DEV smoke test, before the fixed evaluation.
                 Records the implementation hashes the spec asks be frozen.
--stage final    run after tables and figure exist.

Also scans for credentials and machine-specific paths, because the spec requires
no credentials, machine paths, account data, or unrelated internal logs in the
shareable scientific package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}

LEAK_PATTERNS = [
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "huggingface token"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "api key"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "github token"),
    (re.compile(r"ya29\.[A-Za-z0-9_\-]{20,}"), "google oauth token"),
    (re.compile(r"/content/drive/MyDrive"), "user drive path"),
    (re.compile(r"[Cc]:\\\\Users\\\\[^\\\\\"'\s]+"), "windows user path"),
    (re.compile(r"/home/(?!claude)[a-z0-9_.-]+/"), "home directory path"),
]

TEXT_SUFFIXES = {".py", ".md", ".json", ".txt", ".csv", ".jsonl", ".ipynb", ".sh", ".cff", ".yaml", ".yml"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            if p.name == "MANIFEST.json" and p.parent == root:
                continue
            yield p


def scan(path: Path):
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    if path.name == Path(__file__).name:
        return []  # this file defines the patterns; it would always match itself
    try:
        text = path.read_text(encoding="utf8", errors="ignore")
    except OSError:
        return []
    hits = []
    for pattern, label in LEAK_PATTERNS:
        for m in pattern.finditer(text):
            hits.append(dict(kind=label, line=text[:m.start()].count("\n") + 1))
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=("frozen", "final"), required=True)
    p.add_argument("--allow-leaks", action="store_true",
                   help="record findings without failing (review manually)")
    args = p.parse_args()

    root = Path(args.root).resolve()
    files, leaks = {}, {}
    for path in walk(root):
        rel = path.relative_to(root).as_posix()
        files[rel] = dict(bytes=path.stat().st_size, sha256=sha(path))
        found = scan(path)
        if found:
            leaks[rel] = found

    expected_final = ["RESULTS.md", "COMMANDS.md", "figure/comparison.png",
                      "tables/headline_results.csv", "tables/per_scene_results.csv",
                      "tables/per_root_results.csv", "tables/timing_memory.csv",
                      "external_reproduction/commit.txt",
                      "external_reproduction/stdout.txt"]
    missing = [f for f in expected_final if f not in files] if args.stage == "final" else []

    manifest = dict(
        schema="UPDATE_METHOD_RESULTS_MANIFEST_V1",
        stage=args.stage,
        generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        file_count=len(files),
        files=files,
        leak_scan=dict(clean=not leaks, findings=leaks),
        missing_expected_outputs=missing,
        status="PASS" if (not leaks or args.allow_leaks) and not missing else "FAIL",
    )

    out = Path(args.output)
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("stage", "file_count", "missing_expected_outputs", "status")}, indent=2))
    if leaks and not args.allow_leaks:
        print("\nLEAK SCAN FAILED -- do not share this package:")
        print(json.dumps(leaks, indent=2))
        return 1
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
