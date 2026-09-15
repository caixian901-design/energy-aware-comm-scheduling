"""Markdown manuscript -> PDF, with no external toolchain.

Word's HTML importer chokes on a 1.4 MB self-contained HTML file (base64 images),
so the PDF is built directly with reportlab.  Everything the manuscript needs is
supported:

    # / ## / ###      headings
    paragraphs        justified, with **bold**, *italic*, `code`
    $$ ... $$         display equations, rendered by matplotlib mathtext
    ![cap](file.png)  figures, auto-scaled to the text width, with a caption
    | a | b |         pipe tables, styled
    ``` ... ```       code blocks
    ---               horizontal rule
    - item / 1. item  bullet and numbered lists

Usage:
    python tools/build_pdf.py [manuscript.md] [out.pdf] [figure_dir]
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image,
                               KeepTogether, PageBreak, PageTemplate, Paragraph,
                               Spacer, Table, TableStyle)

# ---------------------------------------------------------------------------
# Fonts: register real TrueType faces so Greek letters and arrows survive.
# ---------------------------------------------------------------------------
FONT_DIR = r"C:\Windows\Fonts"
FONT_CANDIDATES = [
    # (reportlab name, filename, bold variant)
    ("Body", "times.ttf", "timesbd.ttf"),
    ("Head", "calibri.ttf", "calibrib.ttf"),
    ("Mono", "consola.ttf", "consolab.ttf"),
]


def register_fonts() -> tuple[str, str, str]:
    body, head, mono = "Times-Roman", "Helvetica-Bold", "Courier"
    for name, reg, bold in FONT_CANDIDATES:
        rp, bp = os.path.join(FONT_DIR, reg), os.path.join(FONT_DIR, bold)
        if not (os.path.exists(rp) and os.path.exists(bp)):
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, rp))
            pdfmetrics.registerFont(TTFont(name + "-Bold", bp))
            pdfmetrics.registerFontFamily(name, normal=name, bold=name + "-Bold",
                                          italic=name, boldItalic=name + "-Bold")
        except Exception as exc:
            print(f"  ! font {reg} failed: {exc}")
            continue
        if name == "Body":
            body = "Body"
        elif name == "Head":
            head = "Head-Bold"
        else:
            mono = "Mono"
    return body, head, mono


BODY, HEAD, MONO = register_fonts()

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
ACCENT = colors.HexColor("#12314f")
RULE = colors.HexColor("#c8d4e0")
TH_BG = colors.HexColor("#eef2f8")
TD_ALT = colors.HexColor("#f7f9fc")

S = {
    "title": ParagraphStyle("title", fontName=HEAD, fontSize=17, leading=21,
                            textColor=ACCENT, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=HEAD, fontSize=13, leading=16,
                         textColor=ACCENT, spaceBefore=14, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName=HEAD, fontSize=11.5, leading=14,
                         textColor=colors.HexColor("#1f4e79"),
                         spaceBefore=10, spaceAfter=3),
    "body": ParagraphStyle("body", fontName=BODY, fontSize=10, leading=14.2,
                           alignment=TA_JUSTIFY, spaceAfter=5),
    "center": ParagraphStyle("center", fontName=BODY, fontSize=10, leading=14.2,
                             alignment=TA_CENTER, spaceAfter=5),
    "caption": ParagraphStyle("caption", fontName=BODY, fontSize=8.4, leading=11,
                              alignment=TA_LEFT, textColor=colors.HexColor("#333333"),
                              spaceBefore=3, spaceAfter=10),
    "list": ParagraphStyle("list", fontName=BODY, fontSize=10, leading=14,
                           alignment=TA_JUSTIFY, leftIndent=14, bulletIndent=4,
                           bulletFontName=BODY, bulletFontSize=10,
                           spaceAfter=2.5),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=8.2, leading=11,
                           backColor=colors.HexColor("#f7f9fc"),
                           borderColor=RULE, borderWidth=0.5, borderPadding=5,
                           bulletFontName=MONO, spaceBefore=4, spaceAfter=8),
    "eqtext": ParagraphStyle("eqtext", fontName=MONO, fontSize=9, leading=12,
                             alignment=TA_CENTER, spaceBefore=4, spaceAfter=6),
}

TEXT_W = A4[0] - 3.6 * cm          # frame width


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
def escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))


# Characters the registered TTFs carry; anything else is transliterated so that
# no glyph ever renders as a black box.
TRANSLIT = {
    "→": "->", "←": "<-", "≥": ">=", "≤": "<=", "≈": "~", "×": "x",
    "−": "-", "–": "-", "—": "--", "·": ".", "≡": "=", "µ": "u",
    "Δ": "d", "∑": "sum", "√": "sqrt", "°": " deg", "∝": " prop ",
    "∞": "inf", "∈": " in ", "±": "+/-", "\u00a0": " ",
    "ᵢ": "i", "₂": "2", "₁": "1", "₀": "0",
}

# CJK: the registered Latin TTFs carry no CJK glyphs, and mixing a second font
# mid-paragraph costs more than it is worth in an English manuscript.  The one
# policy name that appears is transliterated instead.
CJK = {
    "东数西算": "Dong Shu Xi Suan",
}


def clean(text: str) -> str:
    for a, b in CJK.items():
        text = text.replace(a, b)
    for a, b in TRANSLIT.items():
        text = text.replace(a, b)
    # anything CJK that slipped through: drop rather than draw a black box
    return re.sub(r"[\u3000-\u9fff\uff00-\uffef]", "", text)


def inline(text: str) -> str:
    """Markdown inline formatting -> reportlab mini-HTML."""
    text = escape(clean(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="%s" size="8.6">\1</font>' % MONO, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # leftover inline math is rendered as plain text, so strip the delimiters and
    # reduce the most common LaTeX fragments to readable ASCII
    text = re.sub(r"\$([^$]+)\$", lambda m: inline_math(m.group(1)), text)
    return text


def inline_math(tex: str) -> str:
    """Reduce an inline LaTeX snippet to readable plain text."""
    s = tex.strip()
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\(?:mathrm|text|mathit)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|big[lr]?)\s*", "", s)
    s = s.replace(r"\_", "_").replace(r"\,", " ").replace(r"\;", " ")
    s = s.replace(r"\times", " x ").replace(r"\cdot", " . ")
    s = re.sub(r"[{}]", "", s)
    return escape(s)


def render_equation(tex: str, path: str, fontsize: int = 13) -> tuple[str, float]:
    """Render a display equation to PNG; return (path, width_in_points)."""
    body = tex.strip()
    if body.startswith("$$") and body.endswith("$$"):
        body = body[2:-2]
    body = body.strip()
    body = re.sub(r"\\underbrace\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", body)
    body = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", body)
    body = body.replace(r"\qquad", r"\ \ \ ")
    body = re.sub(r"\\big[lr]?([()\[\]])", r"\1", body)

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${body}$", fontsize=fontsize)
    fig.savefig(path, dpi=340, bbox_inches="tight", pad_inches=0.02,
                transparent=False, facecolor="white")
    plt.close(fig)
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    return path, w_px


def fit_width(path: str, max_w: float) -> tuple[float, float]:
    """Scale an image to the text width, preserving aspect ratio."""
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(1.0, max_w / w)
    return w * scale, h * scale


# ---------------------------------------------------------------------------
# Block parser
# ---------------------------------------------------------------------------
def parse_blocks(md: str):
    """Yield (kind, payload) blocks from the Markdown source."""
    lines = md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            yield "code", "\n".join(buf)
            continue

        # display equation: either "$$ ... $$" on one line, or a fenced block
        # whose opening and closing "$$" are alone on their own lines
        if s.startswith("$$"):
            if s.endswith("$$") and len(s) > 4:
                yield "eq", s
                i += 1
            else:
                buf = [s]
                i += 1
                while i < n:
                    buf.append(lines[i])
                    if lines[i].strip().endswith("$$"):
                        i += 1
                        break
                    i += 1
                yield "eq", "\n".join(buf)
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            yield "h", (len(m.group(1)), m.group(2))
            i += 1
            continue

        if re.match(r"^-{3,}$", s):
            yield "hr", None
            i += 1
            continue

        # figure on its own: ![caption](path)
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m:
            yield "fig", (m.group(1), m.group(2))
            i += 1
            continue

        # pipe table
        if s.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            header, body = rows[0], [r for r in rows[2:]]
            yield "table", (header, body)
            continue

        # lists
        if re.match(r"^([-*]|\d+\.)\s+", s):
            items = []
            while i < n and re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()):
                raw = re.sub(r"^([-*]|\d+\.)\s+", "", lines[i].strip())
                items.append(raw)
                i += 1
            yield "list", items
            continue

        # paragraph: join until blank line or another block starter
        buf = [s]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("|", "#", "```", "$$", "!["))
                    or re.match(r"^([-*]|\d+\.)\s+", nxt)
                    or re.match(r"^-{3,}$", nxt)):
                break
            buf.append(nxt)
            i += 1
        yield "p", " ".join(buf)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(md_path: str, pdf_path: str, fig_dir: str) -> int:
    with open(md_path, encoding="utf-8") as fh:
        md = fh.read()

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(pdf_path)), "_eq")
    os.makedirs(tmp_dir, exist_ok=True)

    doc = BaseDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                          topMargin=1.9 * cm, bottomMargin=1.8 * cm,
                          title="Communication scheduling is an energy decision",
                          author="Xian Cai",
                          subject="Makespan-carbon trade-offs for distributed AI training")

    def on_page(canv, d):
        canv.saveState()
        canv.setFont(BODY, 7.4)
        canv.setFillColor(colors.HexColor("#777777"))
        canv.drawString(1.8 * cm, A4[1] - 1.25 * cm,
                        "Communication scheduling is an energy decision - X. Cai, 2026")
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(1.8 * cm, A4[1] - 1.38 * cm, A4[0] - 1.8 * cm, A4[1] - 1.38 * cm)
        canv.line(1.8 * cm, 1.42 * cm, A4[0] - 1.8 * cm, 1.42 * cm)
        canv.drawRightString(A4[0] - 1.8 * cm, 1.15 * cm, f"{d.page}")
        canv.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])

    story: list = []
    eq_n = 0
    fig_n = 0
    story_break_done = [False]

    for kind, payload in parse_blocks(md):
        if kind == "h":
            level, text = payload
            if level == 2 and text.strip().lower() == "abstract" and not story_break_done[0]:
                story.append(PageBreak())
                story_break_done[0] = True
            style = {1: S["title"], 2: S["h2"], 3: S["h3"], 4: S["h3"]}[min(level, 4)]
            story.append(Paragraph(inline(text), style))
            if level == 1:
                story.append(HRFlowable(width="100%", thickness=1.1, color=ACCENT,
                                        spaceBefore=2, spaceAfter=8))
            elif level == 2:
                story.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                        spaceBefore=1, spaceAfter=5))

        elif kind == "p":
            story.append(Paragraph(inline(payload), S["body"]))

        elif kind == "center":
            story.append(Paragraph(inline(payload), S["center"]))

        elif kind == "eq":
            eq_n += 1
            png = os.path.join(tmp_dir, f"eq{eq_n:02d}.png")
            try:
                _, w_px = render_equation(payload, png)
                w, h = fit_width(png, min(TEXT_W * 0.94, 460))
                story.append(Spacer(1, 2))
                story.append(Image(png, width=w, height=h))
                story.append(Spacer(1, 6))
            except Exception as exc:
                print(f"  ! equation {eq_n} failed: {str(exc)[:90]}")
                inner = payload.strip()
                inner = inner[2:] if inner.startswith("$$") else inner
                inner = inner[:-2] if inner.endswith("$$") else inner
                story.append(Paragraph(escape(inner.strip()), S["eqtext"]))

        elif kind == "fig":
            caption, raw = payload
            cand = None
            for c in (raw, os.path.join(fig_dir, raw),
                      os.path.join(os.path.dirname(os.path.abspath(md_path)), raw)):
                if os.path.exists(c):
                    cand = c
                    break
            if not cand:
                print(f"  ! missing figure {raw}")
                continue
            fig_n += 1
            w, h = fit_width(cand, TEXT_W)
            cap = Paragraph(inline(caption), S["caption"]) if caption.strip() else None
            block = [Image(cand, width=w, height=h)]
            if cap is not None and not caption.lower().startswith("figure"):
                block.append(cap)
            story.append(KeepTogether(block))
            if cap is not None and caption.lower().startswith("figure"):
                story.append(cap)

        elif kind == "table":
            header, body = payload
            data = [[Paragraph(inline(c), S["caption"]) for c in header]]
            data += [[Paragraph(inline(c), S["caption"]) for c in r] for r in body]
            ncol = max(len(r) for r in data)
            for r in data:
                while len(r) < ncol:
                    r.append(Paragraph("", S["caption"]))
            # first column narrower for wide tables, wider for short ones
            if ncol >= 9:
                first = TEXT_W * 0.10
            elif ncol >= 5:
                first = TEXT_W * 0.30
            else:
                first = TEXT_W * 0.36
            rest = (TEXT_W - first) / (ncol - 1) if ncol > 1 else TEXT_W
            widths = [first] + [rest] * (ncol - 1)
            t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            style = [
                ("BACKGROUND", (0, 0), (-1, 0), TH_BG),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3d0de")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9bb0c7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
            ]
            for r in range(1, len(data)):
                if r % 2 == 0:
                    style.append(("BACKGROUND", (0, r), (-1, r), TD_ALT))
            t.setStyle(TableStyle(style))
            story.append(Spacer(1, 3))
            story.append(t)
            story.append(Spacer(1, 8))

        elif kind == "list":
            for item in payload:
                story.append(Paragraph(inline(item), S["list"], bulletText="\u2022"))

        elif kind == "code":
            story.append(Paragraph(escape(clean(payload)), S["code"]))

        elif kind == "hr":
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                    spaceBefore=6, spaceAfter=8))

    doc.build(story)
    return eq_n, fig_n


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else "manuscript.md"
    out = sys.argv[2] if len(sys.argv) > 2 else "PaperC.pdf"
    fdir = sys.argv[3] if len(sys.argv) > 3 else "figures"
    eqs, figs = build(md, out, fdir)
    size = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({size:.0f} KB)  {eqs} equations, {figs} figures")
    print(f"fonts: body={BODY} head={HEAD} mono={MONO}")
