"""Publication assets stay source-identical and existing figure links stay valid."""
import hashlib
import importlib.util
import json
import re
from html.parser import HTMLParser
from zipfile import ZipFile

import pytest

from repro.common import ROOT

COLLECTION = ROOT / "figures" / "redesigned"
MANIFEST = json.loads((COLLECTION / "MANIFEST.json").read_text(encoding="utf-8"))


def test_complete_reviewed_collection():
    assert MANIFEST["figure_count"] == len(MANIFEST["figures"]) == 16
    assert MANIFEST["original_assets_covered"] == 20
    assert [item["order"] for item in MANIFEST["figures"]] == list(range(1, 17))
    assert {path.name for path in COLLECTION.iterdir()} == set(MANIFEST["files"]) | {"MANIFEST.json"}
    for item in MANIFEST["figures"]:
        for extension in ("pdf", "svg", "png", "zoom.png", "gray.png"):
            assert item["stem"] + "." + extension in MANIFEST["files"]
    for name, expected in MANIFEST["files"].items():
        content = (COLLECTION / name).read_bytes()
        assert len(content) == expected["bytes"], name
        assert hashlib.sha256(content).hexdigest() == expected["sha256"], name


def test_legacy_figure_paths_are_replaced_not_left_stale():
    assert len(MANIFEST["legacy_aliases"]) == 40
    assert len(set(MANIFEST["legacy_aliases"].values())) == 32
    for legacy, canonical in MANIFEST["legacy_aliases"].items():
        assert (ROOT / legacy).read_bytes() == (COLLECTION / canonical).read_bytes(), legacy


def test_offline_gallery_and_github_gallery_links():
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
            self.cards = 0

        def handle_starttag(self, tag, attrs):
            self.cards += tag == "article"
            self.links.extend(value for key, value in attrs if key in {"src", "href"})

    page = (COLLECTION / "START_HERE.html").read_text(encoding="utf-8")
    parser = Links()
    parser.feed(page)
    assert parser.cards == 16
    for link in parser.links:
        if link.startswith("#"):
            assert f'id="{link[1:]}"' in page
        else:
            assert link in MANIFEST["files"], link
    readme = (COLLECTION / "README.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^!\[", readme, re.M)) == 16
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        assert (COLLECTION / target).is_file(), target


def test_viewing_export_is_complete_and_non_destructive(tmp_path):
    spec = importlib.util.spec_from_file_location("redesigned_packager", ROOT / "scripts/package_redesigned_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "figures.zip"
    report = module.package(output)
    assert report["figures"] == 16
    with ZipFile(output) as archive:
        prefix = "geometry-of-endorsement-figures/"
        assert set(archive.namelist()) == {prefix + name for name in MANIFEST["files"]} | {prefix + "MANIFEST.json"}
        for name in MANIFEST["files"]:
            assert archive.read(prefix + name) == (COLLECTION / name).read_bytes()
    with pytest.raises(FileExistsError):
        module.package(output)
