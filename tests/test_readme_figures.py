"""The README's figures and tables regenerate from their data, and the data agree with the code."""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "tools" / "readme_figures"
ASSETS = ROOT / "docs" / "assets"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"readme_figures_{name}", FIGURES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(name: str):
    return json.loads((FIGURES / name).read_text())


def test_methods_json_matches_the_registry_and_the_protocol():
    from afrbench import protocol as P
    from afrlib import AFR_REGISTRY

    methods = _json("methods.json")
    assert {m["key"] for m in methods} == set(AFR_REGISTRY)
    assert {m["key"] for m in methods if not m["evaluated"]} == set(P.HELD)
    assert sorted(m["key"] for m in methods if m["evaluated"]) == P.roster()
    assert len(methods) == 26 and sum(m["evaluated"] for m in methods) == 24
    assert {m["paradigm"] for m in methods} == {"masks", "per_sample", "face", "makeup", "accessory"}


def test_model_zoo_data_gives_the_numbers_the_readme_states():
    zoo = _json("model_zoo.json")
    assert len(zoo) == 45
    assert len({m["backbone"] for m in zoo}) == 25
    assert len({m["loss"] for m in zoo}) == 9
    assert len({m["dataset"] for m in zoo}) == 3
    assert len({(m["backbone"], m["loss"], m["dataset"]) for m in zoo}) == 45


def test_model_zoo_data_matches_frbench():
    frbench = pytest.importorskip("frbench")
    live = sorted((m.backbone, m.loss, m.dataset) for m in frbench.list_models())
    assert live == sorted((m["backbone"], m["loss"], m["dataset"]) for m in _json("model_zoo.json"))


def test_readme_method_table_is_current():
    table = _script("methods_table")
    text = (ROOT / "README.md").read_text()
    region = text.split(table.START, 1)[1].split(table.END, 1)[0]
    assert region.strip() == table.build(_json("methods.json")).strip()


@pytest.mark.parametrize("name, build", [
    ("paradigms.svg", lambda: _script("paradigms").build(_json("methods.json"))),
    ("by_the_numbers.svg", lambda: _script("by_the_numbers").build(_json("model_zoo.json"), _json("methods.json"))),
    ("model_zoo.svg", lambda: _script("model_zoo").build(_json("model_zoo.json"))),
    ("architecture.svg", lambda: _script("architecture").build()),
])
def test_committed_figures_match_their_scripts(name, build):
    assert (ASSETS / name).read_text() == build(), f"run tools/readme_figures to regenerate docs/assets/{name}"


def test_readme_references_existing_assets():
    import re

    text = (ROOT / "README.md").read_text()
    for path in re.findall(r'src="(docs/assets/[^"]+)"', text):
        assert (ROOT / path).is_file(), path


def test_showcase_window_stays_inside_the_photo():
    showcase = _script("make_showcase")
    geometry = {"face_box": [100.0, 120.0, 300.0, 400.0]}
    x, y, side = showcase.display_window(geometry, [50, 40, 600, 700], 1.2)
    assert side == round(280 * 1.2)
    assert 50 <= x and x + side <= 650 and 40 <= y and y + side <= 740
    # A window larger than the photo shrinks to it.
    assert showcase.display_window(geometry, [0, 0, 200, 200], 3.0) == [0, 0, 200]


def test_showcase_valid_rect_of_a_frame_that_overhangs_the_photo():
    showcase = _script("make_showcase")
    record = types.SimpleNamespace(valid_rect=None, original_box=[-159, -111, 1410], size=[1024, 1024])
    assert showcase.valid_rect_of(record) == [159, 111, 1024, 1024]
    inside = types.SimpleNamespace(valid_rect=None, original_box=[60, 0, 1341], size=[1448, 2048])
    assert showcase.valid_rect_of(inside) == [0, 0, 1341, 1341]
    stored = types.SimpleNamespace(valid_rect=[3, 4, 5, 6], original_box=[0, 0, 9], size=[9, 9])
    assert showcase.valid_rect_of(stored) == [3, 4, 5, 6]


def test_showcase_psnr_counts_only_the_photo():
    import torch

    showcase = _script("make_showcase")
    clean = torch.zeros(3, 8, 8)
    assert showcase.psnr_on_valid(clean, clean, [0, 0, 8, 8]) == float("inf")
    changed = clean.clone()
    changed[:, :, :4] += 1.0                                   # one grey level on the left half only
    assert showcase.psnr_on_valid(clean, changed, [4, 0, 4, 8]) == float("inf")
    assert showcase.psnr_on_valid(clean, changed, [0, 0, 8, 8]) == pytest.approx(10 * __import__("math").log10(255 ** 2 / 0.5))


def test_readme_headline_numbers_match_the_data():
    """The badge, the intro and the strip state the counts the data give: 45+ models and 24+ evaluated methods,
    where the plus means more are coming.
    """
    text = (ROOT / "README.md").read_text()
    methods, zoo = _json("methods.json"), _json("model_zoo.json")
    evaluated = sum(m["evaluated"] for m in methods)
    assert f"AFR%20methods-{evaluated}%2B%20evaluated" in text
    assert f"FR%20models-{len(zoo)}%2B" in text
    assert f"implements {len(methods)} published AFR methods" in text
    assert f"evaluates {evaluated} of them so far" in text
    assert f">{evaluated}+<" in (ASSETS / "by_the_numbers.svg").read_text()
    assert f">{len(zoo)}+<" in (ASSETS / "by_the_numbers.svg").read_text()
