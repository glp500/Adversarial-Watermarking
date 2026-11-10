"""Draw the explainer diagrams as SVG and render them to PNG.

    python docs/figures/make_diagrams.py

Writes docs/figures/diagrams/{name}.svg and {name}.png (2x, white background).
Needs ImageMagick with librsvg (`magick`). Numbers in token_step are illustrative;
numbers in watermark_flow come from the detection_eval run (Qwen3.5-2B, article_001, g0).
"""

from __future__ import annotations

import subprocess
from html import escape
from pathlib import Path

OUT = Path(__file__).parent / "diagrams"
FONT = "Noto Sans, Liberation Sans, DejaVu Sans, sans-serif"
INK, MUTED, LINE, PANEL = "#1d2330", "#5a6374", "#c3cad6", "#f3f5f8"
BLUE, BLUE_SOFT = "#0072B2", "#e2eff8"
ORANGE, ORANGE_SOFT = "#c85200", "#fbe8dc"
GREY = "#a3abb8"


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.parts = w, h, []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, fill=PANEL, stroke=LINE, sw=1.5, rx=10, extra=""):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')  # fmt: skip

    def text(self, x, y, s, size=15, color=INK, weight=400, anchor="start", italic=False):
        style = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
                 f'text-anchor="{anchor}"{style}>{escape(s)}</text>')  # fmt: skip

    def lines(self, x, y, rows, size=14, color=MUTED, gap=None, anchor="start", weight=400):
        for i, r in enumerate(rows):
            self.text(x, y + i * (gap or size * 1.45), r, size, color, weight, anchor)

    def arrow(self, pts, color=INK, sw=2, dash=False):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        mk = {INK: "a-ink", BLUE: "a-blue", ORANGE: "a-orange", MUTED: "a-muted"}[color]
        da = ' stroke-dasharray="6 5"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da} '
                 f'marker-end="url(#{mk})"/>')  # fmt: skip

    def chip(self, x, y, label, fill=INK, color="#ffffff", anchor="start"):
        w = 9 + 7.6 * len(label)
        x0 = x - w if anchor == "end" else x
        self.rect(x0, y, w, 24, fill=fill, stroke=fill, rx=12)
        self.text(x0 + w / 2, y + 16.5, label, 12.5, color, 700, "middle")

    def title(self, s, sub=None):
        self.text(40, 50, s, 24, INK, 700)
        if sub:
            self.text(40, 80, sub, 15, MUTED)

    def render(self, name: str) -> None:
        markers = "".join(
            f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>'
            for i, c in (("a-ink", INK), ("a-blue", BLUE), ("a-orange", ORANGE), ("a-muted", MUTED))
        )
        hatch = "".join(
            f'<pattern id="h-{n}" width="9" height="9" patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate(45)"><rect width="9" height="9" fill="{soft}"/>'
            f'<rect width="3.2" height="9" fill="{c}" opacity="0.55"/></pattern>'
            for n, c, soft in (("blue", BLUE, BLUE_SOFT), ("orange", ORANGE, ORANGE_SOFT))
        )
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}"><defs>{markers}{hatch}</defs>'
               f'<rect width="100%" height="100%" fill="#ffffff"/>{"".join(self.parts)}</svg>')  # fmt: skip
        OUT.mkdir(exist_ok=True)
        (OUT / f"{name}.svg").write_text(svg)
        subprocess.run(["magick", "-density", "192", "-background", "white", str(OUT / f"{name}.svg"),
                        "-flatten", str(OUT / f"{name}.png")], check=True)  # fmt: skip


def watermark_flow() -> None:
    s = Svg(1320, 470)
    s.title("Where the watermark enters",
            "The source is never changed. A model rewrites it, and SynthID tilts the rewrite's word choices.")  # fmt: skip
    yc, yp, yw = 270, 185, 355  # centre row, plain row, watermarked row
    s.rect(40, yc - 60, 190, 120, fill="#ffffff")
    s.text(135, yc - 18, "Source article", 17, INK, 700, "middle")
    s.lines(135, yc + 8, ["PDF or plain text", "stays unchanged"], 14, MUTED, anchor="middle")
    s.arrow([(230, yc), (288, yc)])
    s.rect(290, yc - 60, 230, 120, fill="#ffffff")
    s.text(405, yc - 18, "Generator LLM", 17, INK, 700, "middle")
    s.lines(
        405,
        yc + 8,
        ["rewrites the article twice,", "same prompt and seed"],
        14,
        MUTED,
        anchor="middle",
    )
    # branches
    s.arrow([(520, yc), (560, yc), (560, yp), (628, yp)], BLUE)
    s.arrow([(520, yc), (560, yc), (560, yw), (628, yw)], ORANGE)
    s.chip(630, yp - 74, "SynthID off", fill=BLUE)
    s.chip(630, yw - 74, "SynthID on + secret key", fill=ORANGE)
    s.rect(630, yp - 42, 220, 84, fill=BLUE_SOFT, stroke=BLUE, sw=2)
    s.text(740, yp - 6, "Plain rewrite", 17, INK, 700, "middle")
    s.text(740, yp + 20, "the control", 14, MUTED, anchor="middle")
    s.rect(630, yw - 42, 220, 84, fill=ORANGE_SOFT, stroke=ORANGE, sw=2)
    s.text(740, yw - 6, "Watermarked rewrite", 17, INK, 700, "middle")
    s.text(740, yw + 20, "reads just as normally", 14, MUTED, anchor="middle")
    s.arrow([(850, yp), (890, yp), (890, yc - 20), (928, yc - 20)], BLUE)
    s.arrow([(850, yw), (890, yw), (890, yc + 20), (928, yc + 20)], ORANGE)
    s.rect(930, yc - 60, 170, 120, fill="#ffffff")
    s.text(1015, yc - 18, "Detector", 17, INK, 700, "middle")
    s.lines(1015, yc + 8, ["same secret key,", "statistical test"], 14, MUTED, anchor="middle")
    s.arrow([(1100, yc - 20), (1140, yc - 20), (1140, yp), (1158, yp)], BLUE)
    s.arrow([(1100, yc + 20), (1140, yc + 20), (1140, yw), (1158, yw)], ORANGE)
    s.rect(1160, yp - 38, 130, 76, fill="#ffffff", stroke=BLUE, sw=2)
    s.text(1225, yp - 6, "not detected", 15, BLUE, 700, "middle")
    s.text(1225, yp + 18, "p = 0.62", 14, MUTED, anchor="middle")
    s.rect(1160, yw - 38, 130, 76, fill="#ffffff", stroke=ORANGE, sw=2)
    s.text(1225, yw - 6, "detected", 15, ORANGE, 700, "middle")
    s.text(1225, yw + 18, "p = 1.9 × 10⁻¹⁷", 14, MUTED, anchor="middle")
    s.text(40, 445, "Example: Qwen3.5-2B, one fictional article, same seed for both rewrites. "
           "Detection counts as positive when p < 0.01.", 13.5, MUTED)  # fmt: skip
    s.render("watermark_flow")


def token_step() -> None:
    """Tournament sampling at one word, simplified to 3 rounds and 8 draws.

    The bits are illustrative but consistent: a word's bit for a round depends only on
    (key, previous 4 tokens, word, round), so the same word has the same bits in every slot.
    The signal-per-word values were simulated with transformers' own
    SynthIDTextWatermarkLogitsProcessor.update_scores (30 rounds, 20,000 random key draws).
    """
    s = Svg(1400, 880)
    s.title("How SynthID picks each word: a tournament",
            "Simplified to 3 rounds and 8 draws. The real watermark uses 30 rounds, computed exactly.")  # fmt: skip
    bits = {"rapidly": (0, 1, 1), "quickly": (1, 0, 1), "soon": (1, 1, 0), "slowly": (0, 0, 1)}
    leaves = ["rapidly", "quickly", "rapidly", "slowly", "soon", "rapidly", "quickly", "rapidly"]
    # context
    s.rect(40, 150, 250, 250, fill="#ffffff")
    s.text(165, 180, "Text so far", 16, INK, 700, "middle")
    s.text(165, 210, "“…the moss spread ___”", 14.5, INK, anchor="middle", italic=True)
    s.text(60, 250, "model's odds", 13, MUTED, 700)
    for k, (w, pr) in enumerate(
        (("rapidly", "30%"), ("quickly", "28%"), ("soon", "12%"), ("other words", "30%"))
    ):
        s.text(60, 276 + k * 26, w, 14.5, INK)
        s.text(270, 276 + k * 26, pr, 14.5, INK, anchor="end")
    s.arrow([(290, 275), (338, 275)])
    # bracket geometry
    cols = [340, 580, 820, 1060]
    bw, bh = 170, 36
    heads = [
        "8 draws from the model",
        "round 1: higher g₁ wins",
        "round 2: higher g₂ wins",
        "round 3: higher g₃ wins",
    ]
    for x, h in zip(cols, heads, strict=True):
        s.text(x, 128, h, 13.5, MUTED, 700)

    def box(x, yc, word, hl, out=False):
        stroke, fill = (ORANGE, ORANGE_SOFT) if out else (LINE, "#ffffff")
        s.rect(x, yc - bh / 2, bw, bh, fill=fill, stroke=stroke, sw=2 if out else 1.5, rx=7)
        s.text(x + 12, yc + 5, word, 15, INK, 700 if out else 400)
        for k, b in enumerate(bits[word]):
            bx = x + bw - 54 + k * 16
            s.rect(bx, yc - 6, 12, 12, fill=ORANGE if b else "#ffffff",
                   stroke=INK if k == hl else GREY, sw=2 if k == hl else 1, rx=2)  # fmt: skip

    def join(x0, y_a, y_b, x1, y_out):
        mid = x0 + bw + (x1 - x0 - bw) / 2
        s.add(f'<path d="M{x0 + bw},{y_a} H{mid} V{y_b} M{x0 + bw},{y_b} H{mid} M{mid},{y_out} H{x1 - 2}" '
              f'fill="none" stroke="{GREY}" stroke-width="1.5"/>')  # fmt: skip

    ys = [160 + i * 50 for i in range(8)]
    for w, y in zip(leaves, ys, strict=True):
        box(cols[0], y, w, hl=0)
    # round 1 (g1): rapidly0 v quickly1 -> quickly; rapidly0 v slowly0 -> tie, coin flip -> rapidly;
    # soon1 v rapidly0 -> soon; quickly1 v rapidly0 -> quickly
    r1 = ["quickly", "rapidly", "soon", "quickly"]
    y1 = [(ys[2 * j] + ys[2 * j + 1]) / 2 for j in range(4)]
    for j, (w, y) in enumerate(zip(r1, y1, strict=True)):
        join(cols[0], ys[2 * j], ys[2 * j + 1], cols[1], y)
        box(cols[1], y, w, hl=1)
    s.text(cols[1] + bw + 8, y1[1] + 22, "tie → coin flip", 12, MUTED)
    r2 = ["rapidly", "soon"]
    y2 = [(y1[0] + y1[1]) / 2, (y1[2] + y1[3]) / 2]
    for j, (w, y) in enumerate(zip(r2, y2, strict=True)):
        join(cols[1], y1[2 * j], y1[2 * j + 1], cols[2], y)
        box(cols[2], y, w, hl=2)
    y3 = (y2[0] + y2[1]) / 2
    join(cols[2], y2[0], y2[1], cols[3], y3)
    box(cols[3], y3, "rapidly", hl=-1, out=True)
    s.text(cols[3], y3 + 44, "written next", 14, ORANGE, 700)
    # bit legend
    s.rect(cols[3], 520, 12, 12, fill=ORANGE, stroke=GREY, sw=1, rx=2)
    s.text(cols[3] + 20, 531, "bit = 1", 13, MUTED)
    s.rect(cols[3] + 90, 520, 12, 12, fill="#ffffff", stroke=GREY, sw=1, rx=2)
    s.text(cols[3] + 110, 531, "bit = 0", 13, MUTED)
    s.text(cols[3], 556, "squares: g₁ g₂ g₃", 13, MUTED)
    s.text(cols[3], 578, "outlined: the bit", 13, MUTED)
    s.text(cols[3], 596, "that decides the round", 13, MUTED)
    # bottom panels
    s.add(f'<line x1="40" y1="620" x2="1360" y2="620" stroke="{LINE}" stroke-width="1.5"/>')
    s.text(40, 656, "Where the bits come from", 16, INK, 700)
    s.lines(40, 684, ["secret key + previous 4 words + candidate word + round", "→ hashed into a fixed random table of 0s and 1s.",
                      "Same inputs, same bit, every time. Without the key,", "the bits look like coin flips."], 14, MUTED)  # fmt: skip
    s.text(520, 656, "What the detector checks", 16, INK, 700)
    s.lines(520, 684, ["Recompute the 30 bits of every written word.", "Normal text: about half are 1.",
                       "Watermarked example: 3,876 of 7,050 are 1 (0.550),", "p = 1.9 × 10⁻¹⁷."], 14, MUTED)  # fmt: skip
    s.text(960, 656, "Signal per word depends on choice", 16, INK, 700)
    s.text(960, 678, "expected share of 1-bits in the picked word, 30 rounds", 12.5, MUTED)
    rows = (("95% on one word", 0.518), ("the moss example", 0.620), ("40 equally likely", 0.678))
    x0, sc = 1130, 1200  # bar starts at 0.5, px per unit above 0.5
    for k, (lab, v) in enumerate(rows):
        y = 706 + k * 34
        s.text(960, y + 13, lab, 14, INK)
        s.add(
            f'<line x1="{x0}" y1="{y - 4}" x2="{x0}" y2="{y + 20}" stroke="{MUTED}" stroke-width="1.5"/>'
        )
        s.rect(x0, y, (v - 0.5) * sc, 16, fill=ORANGE, stroke=ORANGE, sw=0, rx=2)
        s.text(x0 + (v - 0.5) * sc + 8, y + 13, f"{v:.3f}", 13.5, INK, 700)
    s.text(x0, 812, "0.5 = no signal", 12.5, MUTED, anchor="middle")
    s.text(40, 856, "Every draw comes from the model, so the winner is always a word the model might have written. "
           "Averaged over keys, each word keeps its original odds.", 13.5, MUTED)  # fmt: skip
    s.render("token_step")


def conditions() -> None:
    s = Svg(1120, 600)
    s.title("The four conditions", "Two yes-or-no questions about what the receiving model gets.")
    cx, cw, ch = (330, 720), 300, 130
    ry = (170, 380)
    s.text(cx[0] + cw / 2, 150, "Receiver not told", 16, INK, 700, "middle")
    s.text(cx[1] + cw / 2, 150, "Receiver told the detector's verdict", 16, INK, 700, "middle")
    s.text(300, ry[0] + 60, "Plain document", 16, BLUE, 700, "end")
    s.text(300, ry[0] + 82, "SynthID off", 13.5, MUTED, anchor="end")
    s.text(300, ry[1] + 60, "Watermarked document", 16, ORANGE, 700, "end")
    s.text(300, ry[1] + 82, "SynthID on", 13.5, MUTED, anchor="end")
    cells = {
        "C0": (0, 0, "baseline", "ordinary prompt"),
        "C2": (1, 0, "told “no watermark detected”", "no extra instruction"),
        "C1": (0, 1, "watermark, no hint", "ordinary prompt"),
        "C3": (1, 1, "told “watermark detected”", "+ harmless marker instruction"),
    }
    for name, (c, r, l1, l2) in cells.items():
        x, y = cx[c], ry[r]
        color, soft, hn = (BLUE, BLUE_SOFT, "blue") if r == 0 else (ORANGE, ORANGE_SOFT, "orange")
        s.rect(x, y, cw, ch, fill=soft, stroke=color, sw=2)
        if c == 1:  # hatched band = told
            s.rect(x + 1, y + 1, 22, ch - 2, fill=f"url(#h-{hn})", stroke="none", sw=0, rx=9)
        s.text(x + 44, y + 50, name, 30, color, 700)
        s.text(x + 44, y + 82, l1, 14.5, INK, 700)
        s.text(x + 44, y + 106, l2, 14, MUTED)
    # contrasts
    ymid = (ry[0] + ch + ry[1]) / 2
    for c, lab in ((0, "C1 − C0  watermark effect"), (1, "C3 − C2  watermark effect, told")):
        x = cx[c] + cw / 2
        s.arrow([(x, ry[0] + ch + 6), (x, ry[1] - 6)], INK)
        s.text(x + 12, ymid + 5, lab, 13.5, INK, 700)
    for r, lab in ((0, "C2 − C0"), (1, "C3 − C1")):
        y = ry[r] + ch / 2
        s.arrow([(cx[0] + cw + 6, y), (cx[1] - 6, y)], MUTED)
        s.text((cx[0] + cw + cx[1]) / 2, y - 10, lab, 13, MUTED, 700, "middle")
        s.text((cx[0] + cw + cx[1]) / 2, y + 24, "telling", 13, MUTED, anchor="middle")
    s.text(40, 568, "Hatching marks the conditions where the verdict is added to the system prompt. "
           "The marker instruction is “start your answer with [WM-EVAL]”.", 13.5, MUTED)  # fmt: skip
    s.render("conditions")


def pipeline() -> None:
    s = Svg(1120, 1200)
    s.title("How one run flows", "Seven steps. Tags show the milestone from the original plan.")
    X, W = 60, 800
    steps = [
        ("1", "Ingest sources", "PDF or text files become data/raw/*.jsonl", "wm-eval ingest", 90),
        ("2", "Generate a matched pair", "same prompt and seed, SynthID off and on", "M1", 90),
        ("3", "Detect the watermark", "secret key gives a p-value for each document", "M2", 90),
        ("4", "Assign the four conditions", "", "M5 · M7", 180),
        ("5", "Receiver does a task", "llama-server model or sandboxed file agent", "M3 · M8", 90),
        (
            "6",
            "Score the response",
            "sentiment, emotion, contradiction, similarity, length, success",
            "M4",
            90,
        ),
        (
            "7",
            "Compare and report",
            "paired differences, bootstrap over source articles, figures",
            "M6",
            90,
        ),
    ]
    y, gap, mids = 120, 44, []
    for n, title, detail, tag, h in steps:
        s.rect(X, y, W, h, fill="#ffffff")
        s.add(f'<circle cx="{X + 44}" cy="{y + 45}" r="21" fill="{INK}"/>')
        s.text(X + 44, y + 52, n, 18, "#ffffff", 700, "middle")
        s.text(X + 84, y + 40, title, 18, INK, 700)
        if detail:
            s.text(X + 84, y + 66, detail, 14.5, MUTED)
        s.chip(X + W - 16, y + 16, tag, anchor="end")
        if n == "2":  # the two documents
            pass
        if n == "4":
            s.lines(X + 84, y + 70, ["every document pair goes to all four", "C2 and C3 add the detector's verdict", "to the receiver's system prompt"], 14.5, MUTED)  # fmt: skip
            gx, gw, gh = X + 420, 170, 52
            for name, c, r, lab in (("C0", 0, 0, "plain · not told"), ("C2", 1, 0, "plain · told"),
                                    ("C1", 0, 1, "watermarked · not told"), ("C3", 1, 1, "watermarked · told")):  # fmt: skip
                cxp, cyp = gx + c * (gw + 14) - 30, y + 56 + r * (gh + 10)
                color, soft, hn = (
                    (BLUE, BLUE_SOFT, "blue") if r == 0 else (ORANGE, ORANGE_SOFT, "orange")
                )
                s.rect(cxp, cyp, gw, gh, fill=soft, stroke=color, sw=1.8, rx=7)
                if c == 1:
                    s.rect(
                        cxp + 1,
                        cyp + 1,
                        14,
                        gh - 2,
                        fill=f"url(#h-{hn})",
                        stroke="none",
                        sw=0,
                        rx=6,
                    )
                s.text(cxp + 24, cyp + 22, name, 15, color, 700)
                s.text(cxp + 24, cyp + 41, lab, 12.5, INK)
        mids.append(y + h / 2)
        if n != "7":
            s.arrow([(X + W / 2, y + h), (X + W / 2, y + h + gap - 2)])
        y += h + gap
    # storage rail
    rx, top, bot = 930, 120, y - gap
    s.rect(
        rx, top, 160, bot - top, fill=PANEL, stroke=MUTED, sw=1.5, extra='stroke-dasharray="7 6"'
    )
    s.text(rx + 80, top + 40, "results/", 16, INK, 700, "middle")
    s.text(rx + 80, top + 62, "RUN_ID/", 16, INK, 700, "middle")
    s.lines(rx + 80, top + 110, ["every step saves", "its inputs and", "outputs here,", "write-once"], 14, MUTED, anchor="middle")  # fmt: skip
    s.lines(rx + 80, top + 250, ["plus a log of", "every event"], 14, MUTED, anchor="middle")
    s.lines(
        rx + 80, bot - 80, ["--resume", "redoes only", "missing work"], 14, MUTED, anchor="middle"
    )
    for m in mids:
        s.add(
            f'<line x1="{X + W}" y1="{m}" x2="{rx}" y2="{m}" stroke="{GREY}" stroke-width="1.5" stroke-dasharray="3 5"/>'
        )
    s.text(60, y + 10, "On a 6 GB GPU, steps 2 and 3 run first. Then llama-server starts and the run resumes at step 5.", 14, MUTED)  # fmt: skip
    s.render("pipeline")


if __name__ == "__main__":
    watermark_flow()
    token_step()
    conditions()
    pipeline()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
