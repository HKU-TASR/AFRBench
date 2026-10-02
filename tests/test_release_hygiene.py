"""Repository hygiene: no private paths or names, and license notices for every method."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(
    r"/data/zlwang|/home/zlwang|/Users/|10\.21\.|code_references|results_eval|ProtegoPlus|instructor",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".sh", ".json", ".cff"}


def _repository_files():
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / name for name in listed if (ROOT / name).suffix in TEXT_SUFFIXES]


def test_no_private_references():
    this = Path(__file__).resolve()
    offenders = []
    for path in _repository_files():
        if path.resolve() == this or not path.is_file():
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if PRIVATE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, offenders[:20]


def test_every_method_has_a_license_or_notice():
    methods = [p for p in (ROOT / "afrlib" / "afr").iterdir() if p.is_dir() and p.name != "__pycache__"]
    missing = [p.name for p in methods if not ((p / "LICENSE").is_file() or (p / "NOTICE").is_file())]
    assert not missing, missing
    third_party = (ROOT / "THIRD_PARTY.md").read_text()
    assert all(f"`{p.name}`" in third_party for p in methods)
