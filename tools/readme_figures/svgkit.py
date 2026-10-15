"""Small SVG helpers shared by the README figure scripts (standard library only).

Every figure is drawn on its own opaque light card, so it reads the same on GitHub's light and dark themes.
"""
from __future__ import annotations

import colorsys
from typing import List, Tuple
from xml.sax.saxutils import escape

FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
INK = "#1f2328"
MUTED = "#59636e"
CARD = "#ffffff"
EDGE = "#d1d9e0"
BAND = "#f6f8fa"

# One colour per method paradigm (the same hues as the results viewer).
PARADIGM_HUES = {
    "masks": 186,
    "per_sample": 248,
    "face": 12,
    "makeup": 328,
    "accessory": 78,
}


def hsl(hue: float, sat: float, light: float) -> str:
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, light, sat)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Rough rendered width of ``text`` in a Helvetica-like font; good enough to lay out chips."""
    total = 0.0
    for ch in text:
        if ch in "ilI.,:;'|!":
            total += 0.30
        elif ch in "fjrt() -/":
            total += 0.38
        elif ch in "mwMW":
            total += 0.86
        elif ch.isupper() or ch.isdigit():
            total += 0.64
        else:
            total += 0.54
    return total * size * (1.06 if bold else 1.0)


class Svg:
    """A tiny append-only SVG writer."""

    def __init__(self, width: int, height: int, title: str, description: str = "") -> None:
        self.width, self.height = width, height
        self.parts: List[str] = []
        self.title, self.description = title, description

    def raw(self, markup: str) -> None:
        self.parts.append(markup)

    def rect(self, x, y, w, h, fill, stroke=None, radius=0, stroke_width=1.0) -> None:
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" '
            f'fill="{fill}"{stroke_attr}/>'
        )

    def text(self, x, y, content, size=13, fill=INK, bold=False, anchor="start", opacity=None) -> None:
        weight = ' font-weight="600"' if bold else ""
        opacity_attr = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{opacity_attr}{weight} '
            f'text-anchor="{anchor}">{escape(content)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke=EDGE, width=1.0) -> None:
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def circle(self, cx, cy, r, fill) -> None:
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>')

    def render(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="t d" '
            f'font-family="{FONT}">\n<title id="t">{escape(self.title)}</title>\n'
            f'<desc id="d">{escape(self.description)}</desc>\n'
            f'<rect x="0.5" y="0.5" width="{self.width - 1}" height="{self.height - 1}" rx="12" '
            f'fill="{CARD}" stroke="{EDGE}"/>\n'
        )
        return head + "\n".join(self.parts) + "\n</svg>\n"


def flow_chips(
    svg: Svg,
    chips: List[Tuple[str, str, bool]],
    x0: float,
    y0: float,
    max_width: float,
    hue: float,
    height: float = 28,
    gap: float = 8,
    font: float = 13,
) -> float:
    """Draw ``(name, note, hollow)`` chips left to right, wrapping at ``max_width``; return the bottom y.

    ``hollow`` chips (methods the benchmark does not evaluate) get a dashed outline and no fill.
    """
    x, y = x0, y0
    for name, note, hollow in chips:
        width = 22 + text_width(name, font, bold=True) + (6 + text_width(note, font - 2) if note else 0)
        if x + width > x0 + max_width and x > x0:
            x, y = x0, y + height + gap
        fill = "#ffffff" if hollow else hsl(hue, 0.55, 0.94)
        stroke = hsl(hue, 0.5, 0.45)
        dash = ' stroke-dasharray="4 3"' if hollow else ""
        svg.raw(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height}" rx="{height / 2}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"{dash}/>'
        )
        tx = x + 11
        svg.text(tx, y + height / 2 + font * 0.35, name, size=font, fill=hsl(hue, 0.5, 0.2), bold=True)
        if note:
            svg.text(tx + text_width(name, font, bold=True) + 6, y + height / 2 + (font - 2) * 0.35, note,
                     size=font - 2, fill=MUTED)
        x += width + gap
    return y + height
