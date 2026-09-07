"""Validate paper prose against fact contracts, terminology, and readability.

Documents under validation are listed in paper/documents.json. Canonical names
come from paper/TERMINOLOGY.json, which is the source of truth: the Results
document seeded it, but does not override it.

Three gates run over each prose unit.

  FIDELITY     every contracted fact survives, every number is declared, every
               artifact-bound number matches its frozen artifact, no forbidden
               claim appears, qualifications are intact. Failures are errors.
  TERMINOLOGY  canonical names are used, flagged aliases are absent, symbols are
               introduced before use. Failures are errors.
  READABILITY  mechanical warnings only. Never fails a run.

A fourth category sits alongside them. A number may be declared *pending*: the
claim is written down and tracked, but no frozen artifact proves it yet. Pending
declarations are reported in their own category and counted by --coverage. They
never decay into ordinary readability warnings, and under --submission-strict a
pending headline claim is an error.

    python scripts/prose_check.py                      # all documents
    python scripts/prose_check.py --document results
    python scripts/prose_check.py --coverage
    python scripts/prose_check.py --submission-strict  # paper-freeze gate
    python scripts/prose_check.py --emit-comprehension DIR
    python scripts/prose_check.py --grade ANSWERS.json

See paper/README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TERMINOLOGY = PAPER / "TERMINOLOGY.json"
DOCUMENTS = PAPER / "documents.json"

# Numbers inside a proper noun are not reported values.
NAME_MASKS = [
    r"Llama-3\.1-8B-Instruct", r"Llama-3\.1-8B", r"Llama 3\.1", r"Gemma-2-9B-it",
    r"Gemma-2-9B", r"MiniLM-L6-v2", r"all-MiniLM", r"MiniLM", r"`A/B`", r"`1/2`",
    r"\bA/B\b", r"\b1/2\b", r"float16", r"bfloat16", r"SHA-256", r"sha256", r"\bL2\b",
    r"AMPERE\+\+", r"US2016", r"AbstRCT", r"GPT-2",
]

# "It is not an accuracy, a logit, or Cohen's d." A statistic's scale should be
# stated positively; a list of things it is not makes the reader carry the
# negations. Warned, not forbidden: a single contrast is often the clearest
# phrasing, so only lists are flagged.
# Defining a statistic by what it is not - "it is not an accuracy, a logit, or
# Cohen's d" - makes the reader carry the negations, and the scale is nearly
# always stated positively in the same breath. Saying what a control rules out
# is a different thing and stays legal, so the rule only fires when the negated
# noun names a statistic type.
STAT_NOUNS = (
    r"accurac(?:y|ies)|logits?|AUROCs?|effect sizes?|probabilit(?:y|ies)|"
    r"Cohen|log-odds|odds ratios?|p-values?|correlations?|nat shifts?"
)
NEGATION_LIST = re.compile(rf"(?:is|are|was|were) not (?:an? |the )?(?:raw )?(?:{STAT_NOUNS})", re.I)

AMBIGUOUS_REFERENCES = [
    r"\bthis quantity\b", r"\bthe former\b", r"\bthe latter\b", r"\bthis statistic\b",
    r"\bthe above\b", r"\bit does so\b", r"\bthese values\b", r"\bthe same thing\b",
]

PROVENANCE_MARKERS = [
    (r"\b[0-9a-f]{12,}\b", "raw hash or commit id"),
    (r"\.zip\b", "archive filename"),
    (r"\bartifacts/", "repository path"),
    (r"\bscripts/", "repository path"),
    (r"\bsha256\b|\bSHA-256\b", "digest reference"),
    (r"\d+\.\d{8,}", "number quoted to 8+ decimals"),
]

ERROR, WARNING, PENDING = "error", "warning", "pending"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

@dataclass
class Finding:
    gate: str
    level: str
    document: str
    unit: str
    rule: str
    message: str

    def __str__(self) -> str:
        mark = {"error": "FAIL", "warning": "warn", "pending": "PEND"}[self.level]
        return f"  [{mark}] {self.gate}/{self.rule}: {self.message}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    coverage: list[dict] = field(default_factory=list)

    def add(self, *args: Any) -> None:
        self.findings.append(Finding(*args))

    def _level(self, level: str) -> list[Finding]:
        return [f for f in self.findings if f.level == level]

    @property
    def errors(self) -> list[Finding]:
        return self._level(ERROR)

    @property
    def warnings(self) -> list[Finding]:
        return self._level(WARNING)

    @property
    def pendings(self) -> list[Finding]:
        return self._level(PENDING)

    def ok(self) -> bool:
        return not self.errors


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(data: dict, dotted: str) -> Any:
    current: Any = data
    for key in dotted.split("."):
        current = current[int(key)] if key.isdigit() and isinstance(current, list) else current[key]
    return current


def parse_units(markdown: str) -> dict[str, str]:
    """Split a document into its level-2 and level-3 sections."""
    units: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in markdown.split("\n"):
        heading = re.match(r"^(#{2,3})\s+(.*)$", line)
        if heading:
            if current is not None:
                units[current] = "\n".join(buffer).strip()
            current = heading.group(2).strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        units[current] = "\n".join(buffer).strip()
    return units


def find_unit(units: dict[str, str], heading_prefix: str) -> tuple[str, str] | None:
    for heading, body in units.items():
        if heading.startswith(heading_prefix):
            return heading, body
    return None


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

def mask_names(text: str) -> str:
    for pattern in NAME_MASKS:
        text = re.sub(pattern, " ", text)
    return text


def normalize_number(token: str) -> str:
    return token.replace("{,}", "").replace(",", "").rstrip(".")


def number_tokens(text: str) -> list[str]:
    text = mask_names(text)
    text = text.replace("{,}", ",")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", " ", text)
    # subscripts and superscripts index a symbol; they are not reported values
    text = re.sub(r"[_^]\{[^}]*\}", " ", text)
    text = re.sub(r"[_^]\d+", " ", text)
    raw = re.findall(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)", text)
    return [normalize_number(t) for t in raw]


def render(value: float | int, decimals: int | None, thousands: bool) -> str:
    if decimals is None:
        return f"{value:,}" if thousands else f"{value}"
    return f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"


def flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def check_fidelity(doc_id: str, unit_id: str, body: str, contract: dict,
                   artifacts: dict, units: dict[str, str], report: Report,
                   strict: bool) -> dict:
    allowed: set[str] = set()
    flat = flatten(body)
    tokens = number_tokens(body)
    bound = pending = 0
    headline_bound = headline_pending = 0

    for entry in contract.get("numbers", []):
        status = entry.get("status", "verified")
        headline = entry.get("headline", False)

        if status == "verified":
            if "path" not in entry:
                report.add("FIDELITY", ERROR, doc_id, unit_id, "binding-malformed",
                           f"{entry.get('text')!r} is marked verified but declares no artifact path")
                continue
            value = resolve(artifacts, entry["path"])
            expected = render(value, entry.get("decimals"), entry.get("thousands", False))
            allowed.add(normalize_number(expected))
            declared = entry.get("text")
            if declared is not None and normalize_number(declared) != normalize_number(expected):
                report.add("FIDELITY", ERROR, doc_id, unit_id, "number-contract-mismatch",
                           f"contract says {declared!r} for {entry['path']} but the artifact renders {expected!r}")
            if entry.get("required", True) and normalize_number(expected) not in tokens:
                report.add("FIDELITY", ERROR, doc_id, unit_id, "number-missing",
                           f"required value {expected!r} ({entry['path']}) does not appear in the unit")
            bound += 1
            headline_bound += int(headline)

        elif status == "pending":
            text = str(entry["value"])
            allowed.add(normalize_number(text))
            if entry.get("required", True) and normalize_number(text) not in tokens:
                report.add("FIDELITY", ERROR, doc_id, unit_id, "number-missing",
                           f"declared pending value {text!r} does not appear in the unit")
            family = entry.get("claimed_source_family", "UNSPECIFIED")
            reason = entry.get("reason", "no frozen artifact resolved")
            level = ERROR if (strict and headline) else PENDING
            tag = "headline" if headline else "incidental"
            report.add("FIDELITY", level, doc_id, unit_id, "artifact-pending",
                       f"{tag} value {text!r} is unverified (family: {family}) - {reason}")
            pending += 1
            headline_pending += int(headline)
        else:
            report.add("FIDELITY", ERROR, doc_id, unit_id, "binding-malformed",
                       f"unknown status {status!r}")

    for entry in contract.get("literals", []):
        allowed.add(normalize_number(str(entry["text"])))

    for token in tokens:
        if token not in allowed:
            report.add("FIDELITY", ERROR, doc_id, unit_id, "number-undeclared",
                       f"numeric token {token!r} is neither artifact-bound, declared pending, "
                       f"nor listed as a literal")

    for fact in contract.get("required_facts", []):
        destination = fact.get("moved_to")
        if destination:
            found = find_unit(units, destination)
            if found is None:
                report.add("FIDELITY", ERROR, doc_id, unit_id, "fact-destination-missing",
                           f"fact {fact['id']!r} declares moved_to={destination!r}, not a unit here")
                continue
            target = flatten(found[1])
        else:
            target = flat
        patterns = fact.get("any_of", [])
        if patterns and not any(re.search(p, target, re.I | re.S) for p in patterns):
            where = f" (expected in {destination!r})" if destination else ""
            report.add("FIDELITY", ERROR, doc_id, unit_id, "fact-missing",
                       f"{fact['id']}: {fact['description']}{where}")
        for pattern in fact.get("all_of", []):
            if not re.search(pattern, target, re.I | re.S):
                report.add("FIDELITY", ERROR, doc_id, unit_id, "fact-incomplete",
                           f"{fact['id']}: missing required element /{pattern}/")

    for rule in contract.get("forbidden", []):
        if re.search(rule["pattern"], flat, re.I | re.S):
            report.add("FIDELITY", ERROR, doc_id, unit_id, "forbidden-claim",
                       f"{rule['id']}: {rule.get('note', 'forbidden pattern present')}")

    for rule in contract.get("qualifiers", []):
        if not re.search(rule["pattern"], flat, re.I | re.S):
            report.add("FIDELITY", ERROR, doc_id, unit_id, "qualifier-missing",
                       f"{rule['id']}: {rule.get('note', 'required qualification absent')}")

    return {
        "document": doc_id, "unit": unit_id,
        "label": contract.get("coverage_label", unit_id),
        "bound": bound, "pending": pending, "total": bound + pending,
        "headline_bound": headline_bound, "headline_pending": headline_pending,
    }


def strip_canonical(text: str, terminology: dict) -> str:
    """Remove canonical names before searching for aliases.

    'strict test set' contains 'test set'; the longer canonical phrase must not
    be reported as a use of the shorter alias.
    """
    names: list[str] = []
    for entry in terminology["terms"]:
        names.append(entry["canonical"])
        names.extend(entry.get("accepted_variants", []))
    if not names:
        return text
    alternation = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.sub(alternation, " ", text, flags=re.IGNORECASE)


def check_terminology(doc_id: str, unit_id: str, body: str, terminology: dict,
                      report: Report) -> None:
    residue = strip_canonical(flatten(body), terminology)
    for term in terminology["terms"]:
        for alias in term.get("flag_aliases", []):
            if re.search(rf"\b{re.escape(alias)}\b", residue, re.I):
                note = term.get("alias_note") or ""
                report.add("TERMINOLOGY", ERROR, doc_id, unit_id, "alias-used",
                           f"{alias!r} denotes '{term['canonical']}'. Use the canonical name. {note}".strip())


def check_symbol_order(doc_id: str, document: str, terminology: dict, report: Report) -> None:
    for term in terminology["terms"]:
        symbol = term.get("symbol")
        if not symbol:
            continue
        first_symbol = re.search(rf"\${re.escape(symbol)}\$", document)
        if not first_symbol:
            continue
        first_name = re.search(re.escape(term["canonical"]), document, re.I)
        if first_name is None:
            report.add("TERMINOLOGY", ERROR, doc_id, "document", "symbol-undefined",
                       f"symbol ${symbol}$ is used but '{term['canonical']}' never appears")
        elif first_symbol.start() < first_name.start() - 200:
            report.add("TERMINOLOGY", ERROR, doc_id, "document", "symbol-before-definition",
                       f"symbol ${symbol}$ appears before '{term['canonical']}' is introduced")


DOT = "․"


def sentences(body: str) -> list[str]:
    text = re.sub(r"```.*?```", " ", body, flags=re.S)
    text = re.sub(r"^\|.*$", " ", text, flags=re.M)          # tables are not prose
    text = re.sub(r"^>.*$", " ", text, flags=re.M)           # block equations
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d)\.(\d)", rf"\1{DOT}\2", text)
    text = re.sub(r"\b(e\.g|i\.e|cf|vs|No)\.", rf"\1{DOT}", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.replace(DOT, ".").strip() for p in parts if p.strip()]


def standalone_numbers(sentence: str) -> list[str]:
    stripped = re.sub(r"\[[^\]]*\]", " ", sentence)
    stripped = re.sub(r"\([^)]*\)", " ", stripped)
    return number_tokens(stripped)


def check_readability(doc_id: str, unit_id: str, body: str, contract: dict,
                      report: Report) -> None:
    limits = contract.get("readability", {})
    max_words = limits.get("max_sentence_words", 40)
    allow_provenance = limits.get("allow_provenance", False)

    for sentence in sentences(body):
        words = len(sentence.split())
        if words > max_words:
            report.add("READABILITY", WARNING, doc_id, unit_id, "long-sentence",
                       f"{words} words: {sentence[:88]}...")
        if len(re.findall(r"\([^)]{25,}\)", sentence)) >= 2:
            report.add("READABILITY", WARNING, doc_id, unit_id, "multiple-asides",
                       f"two or more long parentheticals: {sentence[:88]}...")
        loose = standalone_numbers(sentence)
        if len(loose) >= 4:
            report.add("READABILITY", WARNING, doc_id, unit_id, "numeric-overload",
                       f"{len(loose)} separate numeric results in one sentence: {sentence[:88]}...")
        if NEGATION_LIST.search(sentence):
            report.add("READABILITY", WARNING, doc_id, unit_id, "negation-list",
                       f"defines a statistic by what it is not; state its scale positively: "
                       f"{sentence[:84]}...")
        for pattern in AMBIGUOUS_REFERENCES:
            if re.search(pattern, sentence, re.I):
                report.add("READABILITY", WARNING, doc_id, unit_id, "ambiguous-reference",
                           f"/{pattern}/ in: {sentence[:88]}...")

    if not allow_provenance:
        for pattern, label in PROVENANCE_MARKERS:
            if re.search(pattern, body):
                report.add("READABILITY", WARNING, doc_id, unit_id, "provenance-in-prose",
                           f"{label} in a reader-facing unit; consider Methods or an appendix")


# --------------------------------------------------------------------------
# blind comprehension
# --------------------------------------------------------------------------

class ComprehensionReader(Protocol):
    """A blind reader. Receives ONLY the prose and the questions."""

    def answer(self, prose: str, questions: list[dict]) -> dict[str, str]:
        ...


class UnavailableReader:
    """Default reader. Refuses rather than inventing answers.

    This repository has no model-evaluation API and adds no network dependency.
    Emit prompts, have a model answer them, then score with --grade.
    """

    def answer(self, prose: str, questions: list[dict]) -> dict[str, str]:
        raise RuntimeError(
            "No blind reader is configured. Use --emit-comprehension, collect answers "
            "from a model, then score with --grade."
        )


def build_prompt(unit_title: str, prose: str, questions: list[dict]) -> dict:
    """Prose and questions only. No contract, no gold answers, no source text."""
    return {
        "instructions": (
            "Read the passage. Answer each question using only what the passage states. "
            "If the passage does not answer a question, reply exactly: NOT STATED."
        ),
        "unit": unit_title,
        "passage": prose,
        "questions": [{"id": q["id"], "question": q["question"]} for q in questions],
        "response_schema": {"answers": {"<question id>": "<free text answer>"}},
    }


def grade(questions: list[dict], answers: dict[str, str]) -> dict:
    rows = []
    for q in questions:
        given = (answers.get(q["id"]) or "").strip()
        hit = bool(given) and any(re.search(p, given, re.I | re.S)
                                  for p in q.get("accept_patterns", []))
        rows.append({"id": q["id"], "critical": q.get("critical", False),
                     "recovered": hit, "answer": given})
    critical = [r for r in rows if r["critical"]]
    return {
        "recovered": sum(r["recovered"] for r in rows),
        "total": len(rows),
        "score": (sum(r["recovered"] for r in rows) / len(rows)) if rows else 0.0,
        "critical_total": len(critical),
        "critical_recovered": sum(r["recovered"] for r in critical),
        "critical_pass": all(r["recovered"] for r in critical),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

ARTIFACT_SOURCES = {
    "figures": "artifacts/figures/figure_data.json",
    "leakage": "artifacts/leakage/results.json",
    "truth": "artifacts/truth/v2_results.json",
    "m1ref": "artifacts/m1/development_reference.json",
    "project": "artifacts/project/status.json",
}


def load_artifacts() -> dict:
    """Every frozen artifact a contract may bind to, namespaced by source.

    A contract path is <source>.<dotted path into that file>, so the artifact a
    number came from is visible in the contract itself.
    """
    index: dict = {}
    for name, relative in ARTIFACT_SOURCES.items():
        path = ROOT / relative
        if path.is_file():
            index[name] = load_json(path)
    return index


def documents(selected: str | None = None) -> list[dict]:
    docs = load_json(DOCUMENTS)["documents"]
    return [d for d in docs if selected is None or d["id"] == selected]


def run(selected: str | None = None, strict: bool = False,
        override_path: Path | None = None) -> tuple[Report, dict[str, dict[str, str]]]:
    terminology = load_json(TERMINOLOGY)
    artifacts = load_artifacts()
    report = Report()
    all_units: dict[str, dict[str, str]] = {}

    for doc in documents(selected):
        path = override_path if override_path is not None else ROOT / doc["path"]
        if not path.is_file():
            report.add("FIDELITY", ERROR, doc["id"], "document", "document-missing",
                       f"{doc['path']} not found")
            continue
        text = path.read_text(encoding="utf-8")
        units = parse_units(text)
        all_units[doc["id"]] = units

        contracts = sorted((ROOT / doc["contracts"]).glob("*.json"))
        if not contracts:
            report.add("FIDELITY", ERROR, doc["id"], "document", "no-contracts",
                       f"no fact contracts in {doc['contracts']}")
            continue

        for contract_path in contracts:
            contract = load_json(contract_path)
            found = find_unit(units, contract["heading_prefix"])
            if found is None:
                report.add("FIDELITY", ERROR, doc["id"], contract["unit_id"], "unit-missing",
                           f"no unit whose heading starts with {contract['heading_prefix']!r}")
                continue
            _, body = found
            row = check_fidelity(doc["id"], contract["unit_id"], body, contract,
                                 artifacts, units, report, strict)
            report.coverage.append(row)
            check_terminology(doc["id"], contract["unit_id"], body, terminology, report)
            check_readability(doc["id"], contract["unit_id"], body, contract, report)

        check_symbol_order(doc["id"], text, terminology, report)

    return report, all_units


def print_coverage(report: Report) -> None:
    print("Artifact coverage of declared empirical claims\n")
    current = None
    for row in report.coverage:
        if row["document"] != current:
            current = row["document"]
            print(f"[{current}]")
        total = row["total"]
        pct = f"{row['bound'] / total:.0%}" if total else " n/a"
        flag = "" if row["pending"] == 0 else f"   {row['pending']} pending"
        print(f"  {row['label']:<46} {row['bound']:>3}/{total:<3} bound  {pct:>4}{flag}")
    bound = sum(r["bound"] for r in report.coverage)
    total = sum(r["total"] for r in report.coverage)
    hb = sum(r["headline_bound"] for r in report.coverage)
    ht = hb + sum(r["headline_pending"] for r in report.coverage)
    print()
    if total:
        print(f"  Overall empirical claims: {bound}/{total} ({bound / total:.0%})")
    if ht:
        print(f"  Headline claims:          {hb}/{ht} ({hb / ht:.0%})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", help="restrict to one document id")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--submission-strict", action="store_true",
                        help="pending headline claims become errors")
    parser.add_argument("--emit-comprehension", metavar="DIR")
    parser.add_argument("--grade", metavar="ANSWERS_JSON")
    args = parser.parse_args()

    report, all_units = run(args.document, strict=args.submission_strict)

    if args.emit_comprehension:
        out = Path(args.emit_comprehension)
        out.mkdir(parents=True, exist_ok=True)
        count = 0
        for doc in documents(args.document):
            units = all_units.get(doc["id"], {})
            for spec_path in sorted((ROOT / doc["comprehension"]).glob("*.json")):
                spec = load_json(spec_path)
                found = find_unit(units, spec["heading_prefix"])
                if found is None:
                    continue
                heading, body = found
                bundle = build_prompt(heading, body, spec["questions"])
                (out / f"{doc['id']}.{spec['unit_id']}.prompt.json").write_text(
                    json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
                count += 1
        print(f"wrote {count} blind-reader prompt bundle(s) to {out}")
        print("Gold answers stay in paper/comprehension/ and are not included.")
        return 0

    if args.grade:
        answers_doc = load_json(Path(args.grade))
        overall = True
        for doc in documents(args.document):
            for spec_path in sorted((ROOT / doc["comprehension"]).glob("*.json")):
                spec = load_json(spec_path)
                key = f"{doc['id']}.{spec['unit_id']}"
                given = answers_doc.get(key, answers_doc.get(spec["unit_id"], {})).get("answers", {})
                result = grade(spec["questions"], given)
                status = "PASS" if result["critical_pass"] else "FAIL"
                overall &= result["critical_pass"]
                print(f"{key}: {result['recovered']}/{result['total']} facts "
                      f"({result['score']:.0%}), critical "
                      f"{result['critical_recovered']}/{result['critical_total']} [{status}]")
                for row in result["rows"]:
                    if not row["recovered"]:
                        print(f"    {'CRITICAL' if row['critical'] else 'missed'}: {row['id']}")
        return 0 if overall else 1

    if args.coverage:
        print_coverage(report)
        return 0

    if args.json:
        print(json.dumps({
            "ok": report.ok(),
            "strict": args.submission_strict,
            "errors": [f.__dict__ for f in report.errors],
            "pending": [f.__dict__ for f in report.pendings],
            "warnings": [f.__dict__ for f in report.warnings],
            "coverage": report.coverage,
        }, indent=2))
        return 0 if report.ok() else 1

    grouped: dict[tuple[str, str], list[Finding]] = {}
    for finding in report.findings:
        grouped.setdefault((finding.document, finding.unit), []).append(finding)
    for key in sorted(grouped):
        print(f"{key[0]} / {key[1]}:")
        for finding in grouped[key]:
            print(finding)

    print()
    print(f"{len(report.errors)} error(s), {len(report.pendings)} pending, "
          f"{len(report.warnings)} warning(s)")
    if report.pendings and not args.submission_strict:
        headline = sum(1 for f in report.pendings if f.message.startswith("headline"))
        print(f"  {headline} pending claim(s) are headline claims; "
              f"--submission-strict would fail on those.")
    if report.ok():
        print("FIDELITY and TERMINOLOGY gates pass.")
    return 0 if report.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
