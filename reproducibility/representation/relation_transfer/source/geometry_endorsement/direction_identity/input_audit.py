from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_io import sha256_file


FORBIDDEN_PATH_MARKERS = ("claim2", "sealed", "confirmatory", "human_review", "review_outcome")
EXPECTED = {
    "llama": {"revision":"0e9e39f249a16976918f6564b8830bc894c89659", "layer":19,
        "direction_sha256":"f44fb897ab2617abd95ca5a2f8e67dd2626dd75099a9cdf3973033b3e1b64c99",
        "probe_bundle_sha256":"d1e57f79cd93e7ae7a4db44076dcbe6785c685c31e908e717012c0f509f9ac53"},
    "gemma": {"revision":"11c9b309abf73637e4b6f9a3fa1e92e615547819", "layer":27,
        "direction_sha256":"81c704de24ffc90b468f13bbf494638804f7fafa4b24f2f51d6bb64d2dc1c4e7",
        "probe_bundle_sha256":"bb155a092e3c33b04d6031975c2bbf43f2f57513188451022193d2ab8a9db3d8"},
    "prompt_contract_sha256":"33c046d3724f8a2e2a0f985bdba4edae47ee5baba8343c1da861c2fef63d54da",
}


def assert_safe_path(path: str | Path) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
        raise RuntimeError(f"CLAIM2_ACCESS_BLOCKED:{path}")


def load_json_safe(path: str | Path) -> dict[str, Any]:
    assert_safe_path(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_frozen_inputs(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    files = {
        "project_status": root / "artifacts/project/status.json",
        "project_manifest": root / "artifacts/project/manifest.json",
        "m1_reference": root / "artifacts/m1/development_reference.json",
        "m1_manifest": root / "artifacts/m1/manifest.json",
        "causal_scales": root / "configs/causal_token_smoke_scales.json",
    }
    values = {name: load_json_safe(path) for name,path in files.items()}
    status, reference, scales = values["project_status"], values["m1_reference"], values["causal_scales"]
    mismatches=[]
    models=scales["primary"]["models"]
    for name, expected in (("llama",EXPECTED["llama"]),("gemma",EXPECTED["gemma"])):
        observed=models[name]
        for key in ("model_revision","layer","direction_sha256","probe_bundle_sha256"):
            expected_key="revision" if key=="model_revision" else key
            if observed[key] != expected[expected_key]: mismatches.append(f"{name}.{key}")
        if observed["prompt_contract_sha256"] != EXPECTED["prompt_contract_sha256"]: mismatches.append(f"{name}.prompt_contract_sha256")
    if reference["selected_layer"] != EXPECTED["llama"]["layer"]: mismatches.append("m1_reference.selected_layer")
    if reference["model"]["revision"] != EXPECTED["llama"]["revision"]: mismatches.append("m1_reference.revision")
    if reference["prompt_contract_sha256"] != EXPECTED["prompt_contract_sha256"]: mismatches.append("m1_reference.prompt")
    if status["gemma_moral_relation_development"]["selected_layer"] != EXPECTED["gemma"]["layer"]: mismatches.append("status.gemma.layer")
    for manifest_name, target_name, public_key in (("project_manifest","project_status","status.json"),("m1_manifest","m1_reference","development_reference.json")):
        if values[manifest_name]["public_files"][public_key] != sha256_file(files[target_name]): mismatches.append(f"{manifest_name}.hash")
    return {"status":"FROZEN_INPUTS_VERIFIED" if not mismatches else "FROZEN_INPUT_MISMATCH",
        "mismatches":mismatches, "files":{name:{"absolute_path":str(path.resolve()),"sha256":sha256_file(path)} for name,path in files.items()},
        "models":models, "activation_position":"final prompt token before candidate answer sequence",
        "vector_bytes_local":False,
        "colab_vector_locators": {
            "llama":"external-artifacts/llama/m1_probe_parameters.npz",
            "gemma":"external-artifacts/gemma/m1_probe_parameters.npz"},
        "runtime_requirement":"AUDIT_ONLY must hash the located NPZ member before extraction; absence or mismatch stops."}


def scan_source_firewall(repo_root: str | Path, relative_roots: tuple[str,...]) -> list[str]:
    violations=[]; root=Path(repo_root)
    for relative in relative_roots:
        path=root/relative
        candidates=[path] if path.is_file() else list(path.rglob("*")) if path.exists() else []
        for candidate in candidates:
            if candidate.is_file() and any(marker in candidate.name.lower() for marker in FORBIDDEN_PATH_MARKERS):
                violations.append(str(candidate))
    return violations
