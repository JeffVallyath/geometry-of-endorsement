"""Closed-inventory study exports, with no model weights or credential files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError("Unsafe archive member")
    return path


def create(root, output, capsule_hash, quiescent=False):
    root = root.resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError("Export archive must not be inside its input tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("No symlinks in scientific exports")
            if not path.is_file():
                continue
            name = path.relative_to(root).as_posix()
            safe_name(name)
            if path.suffix.lower() in (".pem", ".key") or path.name.lower() in ("token", "credentials", ".env"):
                raise ValueError("Credential-shaped file in export tree")
            raw = path.read_bytes()
            files[name] = {"sha256": digest(raw), "bytes": len(raw)}
            archive.writestr(name, raw)
        manifest = {"schema": "ALIGNMENT_HISTORY_STAGE1_EXPORT_V1", "capsule_sha256": capsule_hash,
                    "quiescent_snapshot": quiescent, "files": files,
                    "scientific_terminal_present": "scientific/TERMINAL.json" in files}
        archive.writestr("MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2))
    return {"archive_sha256": digest(output.read_bytes()), "files": len(files), **{k: v for k, v in manifest.items() if k != "files"}}


def verify(archive_path, capsule_hash, destination=None):
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(x.endswith("/") for x in names):
            raise ValueError("Duplicate or unexpected directory member")
        for name in names:
            safe_name(name)
        manifest = json.loads(archive.read("MANIFEST.json"))
        if manifest["capsule_sha256"] != capsule_hash or manifest["schema"] != "ALIGNMENT_HISTORY_STAGE1_EXPORT_V1":
            raise ValueError("Wrong scientific capsule or artifact schema")
        if set(names) != set(manifest["files"]) | {"MANIFEST.json"}:
            raise ValueError("Undeclared or missing archive member")
        for name, info in manifest["files"].items():
            member = archive.getinfo(name)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink archive member")
            raw = archive.read(name)
            if len(raw) != info["bytes"] or digest(raw) != info["sha256"]:
                raise ValueError("Artifact member digest mismatch")
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
            for name in names:
                target = destination.joinpath(*safe_name(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(archive.read(name))
    return {"archive_sha256": digest(archive_path.read_bytes()), "files": len(manifest["files"]),
            "capsule_sha256": capsule_hash, "quiescent_snapshot": manifest["quiescent_snapshot"],
            "scientific_terminal_present": manifest["scientific_terminal_present"], "all_member_hashes_verified": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--capsule-hash", required=True)
    p.add_argument("--quiescent", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--extract", type=Path)
    args = p.parse_args()
    result = verify(args.out, args.capsule_hash, args.extract) if args.verify else create(args.root, args.out, args.capsule_hash, args.quiescent)
    print(json.dumps(result))
