from __future__ import annotations

from typing import Mapping


def dataset_audit_markdown(records: list[Mapping], selected: tuple[str,str],
                           primary: Mapping | None = None) -> str:
    lines=["# RELATION_DIRECTION_IDENTITY_V1 dataset audit","",
        "This is metadata-only setup evidence. No model was loaded and no scientific result was generated.","",
        f"Frozen secondary datasets: `{selected[0]}` and `{selected[1]}`.",""]
    if primary:
        lines.extend(["## Primary dataset: AMPERE++","",
            f"- Official source/citation: {primary['citation']}",
            f"- Release/license: {primary['version']}; {primary['license']} ({primary['license_source']})",
            f"- Archive SHA-256: `{primary['archive_sha256']}`; publisher MD5: `{primary['publisher_md5']}`",
            f"- Extracted tree SHA-256: `{primary['tree_sha256']}` ({primary['file_count']} files)",
            f"- Orientation/labels: {primary['orientation']}",
            f"- Authoritative splits: {primary['split_counts']}",
            f"- Group unit/count: {primary['group_unit']}; {primary['group_count']} groups",
            f"- Usable binary pairs: {primary['usable_pairs']} ({primary['class_counts']})",
            f"- Pair sufficiency: {primary['pair_sufficiency']}",
            f"- Redistribution: {primary['redistribution']}",
            f"- Local restricted-text path: `{primary['local_path']}` (outside Git)",
            "- Disposition: `PRIMARY_DATASET_READY`; every load-bearing field was verified before selection.",""])
    lines.extend(["## Official ARIES secondary-candidate audit","",
        "| Dataset | Domain | License | Support | Attack | Predefined train/dev/test | Eligible | Exclusions |","|---|---|---:|---:|---:|---:|---:|---|"])
    for row in records:
        failures=", ".join(row["eligibility_failures"]) or "-"
        lines.append(f"| {row['dataset_id']} | {row['domain']} | {row['license_classification']} | {row['support_count']} | {row['attack_count']} | {row['predefined_train_dev_test']} | {row['eligible']} | {failures} |")
    lines.extend(["","AbstRCT's official BRAT schema distinguishes `Attack` and `Partial-Attack`; both are explicit negative argumentative relations and are frozen as `attack`. The authoritative train/dev/test and document IDs are preserved.",
        "","## Claim boundary","",
        "Eligibility is based only on official metadata, licensing, class counts, grouping, and split structure. No dataset was selected using model behavior.",""])
    return "\n".join(lines)


def prompt_contract_markdown(contract_hash: str) -> str:
    return f"""# Frozen prompt contract

Contract SHA-256: `{contract_hash}`

- Extraction position: final prompt position immediately before the candidate answer sequence.
- Templates: `EXPLICIT_RELATION` and structurally distinct `LEXICALLY_ABLATED_RELATION`.
- Mappings: A/B standard and reversed; 1/2 standard and reversed.
- Candidate answers are scored by full sequence probability; single-token status is never assumed.
- Every activation records semantic and physical-token labels independently.
- Candidate token sequences are verified and saved separately for each frozen tokenizer before inference.
"""
