"""Draw the FR model zoo behind AFRBench: 45 models along three axes (architecture, loss, training set).

The model list is ``frbench.list_models()``, stored in ``model_zoo.json`` so the figure builds without the
FR packages. ``--refresh`` re-reads it from an environment where ``frbench`` is installed.

    python tools/readme_figures/model_zoo.py            # writes docs/assets/model_zoo.svg
    python tools/readme_figures/model_zoo.py --refresh  # update model_zoo.json first
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.readme_figures.svgkit import INK, MUTED, Svg, flow_chips, hsl
except ImportError:  # run as a script
    from svgkit import INK, MUTED, Svg, flow_chips, hsl

HERE = Path(__file__).resolve().parent

BACKBONE_NAMES = {
    "irse-18": "IRSE-18", "irse-34": "IRSE-34", "irse-50": "IRSE-50", "irse-100": "IRSE-100",
    "resnet-100": "ResNet-100", "resnetv2-100": "ResNetV2-100", "densenet-201": "DenseNet-201",
    "mobilefacenet": "MobileFaceNet", "mobilenet-w1": "MobileNet", "mobilenetv2-w1": "MobileNetV2",
    "mobilenetv3-l": "MobileNetV3-L", "mobilenetv4conv-m": "MobileNetV4-M", "efficientnetv1-b4": "EfficientNet-B4",
    "convnext-b": "ConvNeXt-B", "convnextv2-b": "ConvNeXtV2-B",
    "swinv1-b": "Swin-B", "swinv2-t": "SwinV2-T", "swinv2-s": "SwinV2-S", "swinv2-b": "SwinV2-B",
    "swinv2-l": "SwinV2-L", "swinmlp-b": "Swin-MLP-B",
    "mobilevit-s": "MobileViT-S", "mobilevitv2-2.0": "MobileViTV2", "mobilevitv3-s": "MobileViTV3-S",
    "mobilevitv3-2.0": "MobileViTV3",
}
# Rows of the architecture panel, in display order.
FAMILIES = [
    ("Classic CNNs", 210, ["irse-18", "irse-34", "irse-50", "irse-100", "resnet-100", "resnetv2-100", "densenet-201"]),
    ("Efficient CNNs", 150, ["mobilefacenet", "mobilenet-w1", "mobilenetv2-w1", "mobilenetv3-l", "mobilenetv4conv-m",
                             "efficientnetv1-b4"]),
    ("Modern CNNs", 32, ["convnext-b", "convnextv2-b"]),
    ("Swin transformers", 280, ["swinv1-b", "swinv2-t", "swinv2-s", "swinv2-b", "swinv2-l", "swinmlp-b"]),
    ("Mobile vision transformers", 340, ["mobilevit-s", "mobilevitv2-2.0", "mobilevitv3-s", "mobilevitv3-2.0"]),
]
LOSS_NAMES = {
    "arcface": "ArcFace", "cosface": "CosFace", "adaface": "AdaFace", "curricularface": "CurricularFace",
    "magface": "MagFace", "sphereface": "SphereFace", "uniface": "UniFace", "centerloss": "Center loss",
    "tripletloss": "Triplet loss",
}
DATASET_NAMES = {"ms1m": "MS1M", "webface4m": "WebFace4M", "glint360k": "Glint360K"}


def refresh(path: Path) -> None:
    import frbench

    rows = [{"key": m.key, "backbone": m.backbone, "loss": m.loss, "dataset": m.dataset}
            for m in frbench.list_models()]
    path.write_text(json.dumps(sorted(rows, key=lambda r: (r["backbone"], r["loss"], r["dataset"])), indent=1) + "\n")
    print(f"wrote {path} ({len(rows)} models)")


def build(zoo: list) -> str:
    """Lay the figure out once to measure it, then draw it on a card of exactly that height."""
    _, bottom = _draw(zoo, 2000)
    return _draw(zoo, int(bottom) + 26)[0].render()


def _draw(zoo: list, height: int):
    backbones = sorted({m["backbone"] for m in zoo})
    placed = [b for _, _, members in FAMILIES for b in members]
    assert sorted(placed) == backbones, sorted(set(backbones) ^ set(placed))
    losses = sorted({m["loss"] for m in zoo})
    datasets = sorted({m["dataset"] for m in zoo})
    assert set(losses) == set(LOSS_NAMES) and set(datasets) == set(DATASET_NAMES)
    # The two backbones trained along the loss and dataset axes.
    deep = sorted({m["backbone"] for m in zoo if (m["loss"], m["dataset"]) != ("arcface", "ms1m")})
    assert deep == ["irse-100", "swinv2-b"], deep
    base = [m for m in zoo if (m["loss"], m["dataset"]) == ("arcface", "ms1m")]
    extra_loss = [m for m in zoo if m["dataset"] == "ms1m" and m["loss"] != "arcface"]
    extra_data = [m for m in zoo if m["loss"] == "arcface" and m["dataset"] != "ms1m"]
    assert len(base) + len(extra_loss) + len(extra_data) == len(zoo)

    width, pad = 960, 22
    left_w, right_x = 560, 624
    svg = Svg(
        width, height, f"The {len(zoo)} face-recognition models of AFRBench",
        f"{len(zoo)} models: {len(backbones)} backbones, {len(losses)} losses and {len(datasets)} training sets.",
    )
    svg.text(pad, 36, f"The {len(zoo)} face-recognition models of FRBench", size=19, bold=True)
    svg.text(pad, 57, f"{len(backbones)} backbones, {len(losses)} loss functions and {len(datasets)} training sets, "
                      "all behind one interface.", size=12.5, fill=MUTED)

    # Architecture panel.
    y = 86
    svg.text(pad, y, f"Architecture: {len(backbones)} backbones, each trained with ArcFace on MS1M", size=14, bold=True)
    y += 16
    for name, hue, members in FAMILIES:
        svg.text(pad, y + 17, name, size=11.5, fill=hsl(hue, 0.5, 0.3), bold=True)
        bottom = flow_chips(svg, [(BACKBONE_NAMES[b], "", False) for b in members], pad, y + 24, left_w - pad, hue,
                            height=24, gap=6, font=12)
        y = bottom + 12

    # Loss and training-set panels.
    x = right_x
    svg.line(x - 34, 78, x - 34, height - 24, stroke="#eaeef2")
    ey = 86
    svg.text(x, ey, f"Loss: {len(losses)} functions", size=14, bold=True)
    svg.text(x, ey + 18, "on IRSE-100 (CNN) and SwinV2-B (transformer)", size=11.5, fill=MUTED)
    ordered = ["arcface", "cosface", "adaface", "curricularface", "magface", "sphereface", "uniface", "centerloss",
               "tripletloss"]
    bottom = flow_chips(svg, [(LOSS_NAMES[l], "", False) for l in ordered], x, ey + 30, width - x - pad, 215,
                        height=24, gap=6, font=12)
    ey = bottom + 26
    svg.text(x, ey, f"Training set: {len(datasets)}", size=14, bold=True)
    svg.text(x, ey + 18, "ArcFace on IRSE-100 and SwinV2-B", size=11.5, fill=MUTED)
    bottom = flow_chips(svg, [(DATASET_NAMES[d], "", False) for d in ["ms1m", "webface4m", "glint360k"]], x, ey + 30,
                        width - x - pad, 160, height=24, gap=6, font=12)

    # How the models add up.
    ey = bottom + 26
    svg.rect(x, ey, width - x - pad, 56, "#f6f8fa", radius=8)
    svg.text(x + 14, ey + 22, f"{len(base)} + {len(extra_loss)} + {len(extra_data)} = {len(zoo)} models", size=13,
             bold=True)
    svg.text(x + 14, ey + 41, f"{len(base)} backbones, {len(extra_loss)} more losses, "
                              f"{len(extra_data)} more training sets", size=10.5, fill=MUTED)
    return svg, max(y, ey + 56)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoo", type=Path, default=HERE / "model_zoo.json")
    parser.add_argument("--out", type=Path, default=HERE.parents[1] / "docs" / "assets" / "model_zoo.svg")
    parser.add_argument("--refresh", action="store_true", help="re-read the model list from frbench first")
    args = parser.parse_args()
    if args.refresh:
        refresh(args.zoo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(json.loads(args.zoo.read_text())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
