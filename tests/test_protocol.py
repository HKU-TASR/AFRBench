"""AFRBench-basic protocol constants: FR-system digests and the method roster."""
from __future__ import annotations

import pytest

from afrbench import protocol
from afrbench.fr_system import FRSystem

# Digests of the FR systems used to produce the AFRBench-basic results. They key the
# embedding caches, so a preset must not change silently.
GOLDEN = {
    "minimal": "3a9efb8b68aa9518",
    "quantized": "820b39bd3fcb6f74",
    "jpeg": "56a2c1cf76524b78",
    "tta": "b3993cfae63412a2",
    "full": "b06e22ec6ac7a262",
    "median": "224fe0ad7f127da3",
    "gfpgan": "5af0cb13305f4a01",
    "codeformer": "e3c8ffd868a1940b",
    "diffpure": "022a3e2523ce5a8b",
}


@pytest.mark.parametrize("preset", sorted(GOLDEN))
def test_preset_digests_are_frozen(preset):
    assert FRSystem.from_dict(protocol.fr_system(preset)).digest == GOLDEN[preset]


def test_suite_systems_match_the_released_results():
    assert FRSystem.from_dict(protocol.jpeg_only_system()).digest == "f3321745349a9dc0"
    whitebox = {
        "whitebox_swinv2b_arcface_glint360k": "75c3f4f0277af804",
        "whitebox_swinv2b_centerloss_ms1m": "5fd866a0aacf7a0b",
        "whitebox_convnextv2b_arcface_ms1m": "7049fcf697692e0f",
    }
    for name, digest in whitebox.items():
        system, visual_quality, distractors = protocol.eval_job(name, dataset=None)
        assert FRSystem.from_dict(system).digest == digest
        assert visual_quality is False and distractors == {"mode": "full"}


def test_eval_suite_names_and_default():
    assert len(protocol.EVAL_SUITE) == 17
    system, visual_quality, distractors = protocol.eval_job("minimal", dataset=None)
    assert system["name"] == "irse100_arcface_ms1m@minimal"
    assert visual_quality is True and distractors == {"mode": "full"}


def test_roster_excludes_held_methods():
    roster = protocol.roster()
    assert len(roster) == 24
    assert not set(roster) & protocol.HELD
    assert protocol.HELD == {"efficient_decision", "dpa"}


def test_fr_slug_for_plugins():
    assert protocol.fr_slug({"backend": "my_pkg.fr:MyRecognizer"}) == "myrecognizer"
