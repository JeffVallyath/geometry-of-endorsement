"""Export the reviewed presentation assets, without regenerating scientific results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "figures" / "redesigned"


def package(output: Path) -> dict:
    manifest_bytes = (COLLECTION / "MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    members = {"MANIFEST.json": manifest_bytes}
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Unsafe figure path: {name}")
        content = (COLLECTION / name).read_bytes()
        if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"Figure collection checksum mismatch: {name}")
        members[name] = content
    output = Path(output).resolve()
    if output.is_relative_to(COLLECTION.resolve()):
        raise ValueError("Place the viewing ZIP outside the committed figure collection")
    with ZipFile(output, "x", ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in members.items():
            archive.writestr("geometry-of-endorsement-figures/" + name, content)
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Viewing ZIP integrity check failed")
    return {"status": "verified_figures_exported", "figures": manifest["figure_count"],
            "files": len(members), "output": str(output),
            "scope": "unchanged presentation assets; no rendering or model inference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New ZIP path; existing files are never overwritten")
    args = parser.parse_args()
    print(json.dumps(package(args.out), indent=2))


if __name__ == "__main__":
    main()
