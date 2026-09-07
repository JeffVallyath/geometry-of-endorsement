"""Search retained archives for the values a pending Results claim reports.

For every experiment family with pending claims, this scans candidate archives
and counts how many of that family's *distinctive* published values reproduce
inside them. Distinctive means a value precise enough that a coincidental match
is unlikely: at least three decimal places, or five significant digits.

An archive is a canonical candidate only when several distinctive values match.
If two archives match comparably, the family is reported AMBIGUOUS and nothing
is bound: guessing between them is exactly the failure this is meant to prevent.

    python scripts/resolve_provenance.py --out paper/PROVENANCE_CANDIDATES.md

Nothing here binds a contract. It produces evidence for a human decision.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "paper" / "contracts" / "results"

ARCHIVE_ROOT_ENV = "GOE_ARCHIVE_ROOTS"  # os.pathsep-separated directories
MAX_ARCHIVE_BYTES = 250 * 1024 * 1024
MAX_MEMBER_BYTES = 25 * 1024 * 1024
TEXT_SUFFIXES = {".json", ".txt", ".csv", ".md", ".yaml", ".yml"}
NAME_HINTS = ("claim2", "relation", "causal", "fingerprint", "semantic", "counterfactual",
              "geometry", "m1", "truth", "human-review", "payoff", "readout", "secb",
              "transfer", "purity", "steering", "intervention")


def distinctive(value: str) -> bool:
    if "." in value:
        return len(value.split(".")[1]) >= 3
    return len(value.replace(",", "").lstrip("-")) >= 5


def pending_by_family() -> dict[str, dict[str, set[str]]]:
    families: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"all": set(), "distinctive": set(), "sections": set()})
    for path in sorted(CONTRACTS.glob("*.json")):
        contract = json.loads(path.read_text(encoding="utf-8"))
        for entry in contract.get("numbers", []):
            if entry.get("status") != "pending" or not entry.get("headline"):
                continue
            family = entry.get("claimed_source_family", "UNSPECIFIED")
            value = str(entry["value"])
            families[family]["all"].add(value)
            families[family]["sections"].add(contract["coverage_label"].split()[0])
            if distinctive(value):
                families[family]["distinctive"].add(value)
    return families


def search_roots() -> list[Path]:
    raw = os.environ.get(ARCHIVE_ROOT_ENV, "")
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()]


def candidate_archives() -> list[Path]:
    found: list[Path] = []
    for root in search_roots():
        if not root.is_dir():
            continue
        for path in root.rglob("*.zip"):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_ARCHIVE_BYTES:
                continue
            if not any(h in path.name.lower() for h in NAME_HINTS):
                continue
            found.append(path)
    return sorted(found)


def archive_text(path: Path) -> str:
    chunks: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.file_size > MAX_MEMBER_BYTES:
                    continue
                if Path(info.filename).suffix.lower() not in TEXT_SUFFIXES:
                    continue
                try:
                    chunks.append(archive.read(info).decode("utf-8", errors="ignore"))
                except Exception:
                    continue
    except Exception:
        return ""
    return "\n".join(chunks)


def matches(text: str, values: set[str]) -> set[str]:
    hit = set()
    stripped = text.replace(",", "")
    for value in values:
        needle = value.replace(",", "")
        if re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d])", stripped):
            hit.add(value)
    return hit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="PROVENANCE_CANDIDATES.md",
                        help="report destination; keep it out of the repository")
    args = parser.parse_args()

    families = pending_by_family()
    archives = candidate_archives()
    if not archives:
        print(f"No archives found. Set {ARCHIVE_ROOT_ENV} to the directories holding them.")
        return 1
    print(f"{len(families)} families with pending headline claims, "
          f"{len(archives)} candidate archives")

    results: dict[str, list[tuple[Path, set[str], set[str]]]] = defaultdict(list)
    for index, archive in enumerate(archives, 1):
        text = archive_text(archive)
        if not text:
            continue
        for family, values in families.items():
            hit_all = matches(text, values["all"])
            hit_distinct = matches(text, values["distinctive"])
            if hit_distinct:
                results[family].append((archive, hit_all, hit_distinct))
        if index % 20 == 0:
            print(f"  scanned {index}/{len(archives)}")

    lines = [
        "# Candidate provenance for pending Results claims",
        "",
        "Generated by `scripts/resolve_provenance.py`. Each row reports how many of a",
        "section's distinctive published values were found inside a candidate archive.",
        "Distinctive means three or more decimal places, or five or more significant",
        "digits, so a coincidental match is unlikely.",
        "",
        "**Nothing here is binding.** An archive becomes canonical only after a human",
        "confirms it, and only when several distinctive values reproduce and the run",
        "metadata agrees. Where two archives match comparably the family is marked",
        "AMBIGUOUS and stays pending.",
        "",
        "**A match means the archive contains the value, not that it produced it.**",
        "Summary bundles, handoff packets and review packages quote results from runs",
        "they did not perform, and they will score highly here. This tool does not yet",
        "check run metadata, so a STRONG verdict narrows the search; it does not close",
        "it. Before promoting a candidate, confirm it holds the run itself: a config",
        "digest, a model revision, and the inputs the numbers were computed from.",
        "",
        "| Family | Sections | Candidate archive | Distinctive matched | All matched | Verdict |",
        "|---|---|---|---:|---:|---|",
    ]
    summary: list[str] = []
    for family in sorted(families):
        values = families[family]
        rows = sorted(results.get(family, []), key=lambda r: -len(r[2]))
        if not rows:
            lines.append(f"| `{family}` | {', '.join(sorted(values['sections']))} | "
                         f"— | 0/{len(values['distinctive'])} | 0/{len(values['all'])} | "
                         f"NO CANDIDATE |")
            summary.append(f"{family}: no candidate archive found")
            continue
        best = len(rows[0][2])
        contenders = [r for r in rows if len(r[2]) >= max(2, best - 1)]
        verdict = "AMBIGUOUS" if len(contenders) > 1 else (
            "STRONG" if best >= 3 else "WEAK")
        for archive, hit_all, hit_distinct in rows[:3]:
            lines.append(
                f"| `{family}` | {', '.join(sorted(values['sections']))} | "
                f"`{archive.name}` | {len(hit_distinct)}/{len(values['distinctive'])} | "
                f"{len(hit_all)}/{len(values['all'])} | "
                f"{verdict if archive is rows[0][0] else 'contender'} |")
        summary.append(f"{family}: {verdict} ({best}/{len(values['distinctive'])} distinctive, "
                       f"{len(contenders)} contender(s))")

    lines += ["", "## Summary", ""]
    lines += [f"- {s}" for s in summary]
    lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\nwrote {args.out}")
    for s in summary:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
