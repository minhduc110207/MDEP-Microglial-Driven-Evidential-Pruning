from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "THEORY_CHANGES.md"
OUT_DIR = ROOT / "output" / "pdf"
OUT_FILE = OUT_DIR / "guds_edl_theory_changes.pdf"


def register_fonts() -> tuple[str, str, str]:
    regular = Path(r"C:\Windows\Fonts\arial.ttf")
    bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    italic = Path(r"C:\Windows\Fonts\ariali.ttf")
    if regular.exists():
        pdfmetrics.registerFont(TTFont("ArialUnicode", str(regular)))
        if bold.exists():
            pdfmetrics.registerFont(TTFont("ArialUnicode-Bold", str(bold)))
        else:
            pdfmetrics.registerFont(TTFont("ArialUnicode-Bold", str(regular)))
        if italic.exists():
            pdfmetrics.registerFont(TTFont("ArialUnicode-Italic", str(italic)))
        else:
            pdfmetrics.registerFont(TTFont("ArialUnicode-Italic", str(regular)))
        return "ArialUnicode", "ArialUnicode-Bold", "ArialUnicode-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


def build_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["Title"] = ParagraphStyle(
        "CustomTitle",
        parent=base["Title"],
        fontName=FONT_BOLD,
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#16324F"),
        spaceAfter=14,
    )
    styles["Subtitle"] = ParagraphStyle(
        "Subtitle",
        fontName=FONT,
        fontSize=10.5,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#3D4F63"),
        spaceAfter=18,
    )
    styles["H1"] = ParagraphStyle(
        "H1",
        fontName=FONT_BOLD,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#16324F"),
        spaceBefore=11,
        spaceAfter=6,
        keepWithNext=True,
    )
    styles["H2"] = ParagraphStyle(
        "H2",
        fontName=FONT_BOLD,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        fontName=FONT,
        fontSize=9.3,
        leading=13.2,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#20262E"),
        spaceAfter=5.2,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=styles["Body"],
        leftIndent=16,
        firstLineIndent=-9,
        bulletIndent=4,
    )
    styles["Formula"] = ParagraphStyle(
        "Formula",
        fontName="Courier",
        fontSize=8.1,
        leading=10.4,
        leftIndent=9,
        rightIndent=9,
        borderPadding=6,
        borderWidth=0.4,
        borderColor=colors.HexColor("#C7D3E0"),
        backColor=colors.HexColor("#F7FAFC"),
        textColor=colors.HexColor("#1C2833"),
        spaceBefore=3,
        spaceAfter=7,
    )
    styles["Small"] = ParagraphStyle(
        "Small",
        fontName=FONT,
        fontSize=8,
        leading=10.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#607080"),
    )
    return styles


STYLES = build_styles()


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", rf'<font name="{FONT_BOLD}">\1</font>', text)
    return text


def code_flowable(code: str):
    cleaned = code.rstrip("\n").replace("\t", "    ")
    max_len = max((len(line) for line in cleaned.splitlines()), default=1)
    # Keep LaTeX commands intact. Instead of hard-wrapping formulas, shrink the
    # monospace font just enough for the longest line to fit the text column.
    usable_width_pt = 7.15 * inch
    font_size = min(8.0, max(5.8, usable_width_pt / max(max_len, 1) / 0.58))
    style = ParagraphStyle(
        f"Formula_{max_len}",
        parent=STYLES["Formula"],
        fontSize=font_size,
        leading=font_size * 1.28,
    )
    return Preformatted(cleaned, style, maxLineLength=10_000, dedent=0)


def paragraph(text: str):
    return Paragraph(inline_markup(text), STYLES["Body"])


def parse_markdown(md: str):
    story = []
    lines = md.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    para: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            text = " ".join(x.strip() for x in para).strip()
            if text:
                story.append(paragraph(text))
            para = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(code_flowable("\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                flush_para()
                in_code = True
                code_lines = []
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped.startswith("# "):
            flush_para()
            title = stripped[2:].strip()
            story.append(Paragraph(inline_markup(title), STYLES["Title"]))
            story.append(
                Paragraph(
                    "Bản ghi chú lý thuyết, công thức và chứng minh liên quan đến GUDS-EDL / MDEP.",
                    STYLES["Subtitle"],
                )
            )
            story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#CBD5E1")))
            story.append(Spacer(1, 8))
            i += 1
            continue

        if stripped.startswith("## "):
            flush_para()
            story.append(Paragraph(inline_markup(stripped[3:].strip()), STYLES["H1"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_para()
            story.append(Paragraph(inline_markup(stripped[4:].strip()), STYLES["H2"]))
            i += 1
            continue

        if stripped.startswith("- "):
            flush_para()
            story.append(
                Paragraph(inline_markup(stripped[2:].strip()), STYLES["Bullet"], bulletText="-")
            )
            i += 1
            continue

        para.append(line)
        i += 1

    flush_para()
    if code_lines:
        story.append(code_flowable("\n".join(code_lines)))
    return story


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = letter
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#607080"))
    canvas.drawString(doc.leftMargin, height - 0.42 * inch, "GUDS-EDL Theory Notes")
    canvas.drawRightString(width - doc.rightMargin, height - 0.42 * inch, "main_text.tex theory changes")
    canvas.setStrokeColor(colors.HexColor("#D5DEE8"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, height - 0.48 * inch, width - doc.rightMargin, height - 0.48 * inch)
    canvas.drawCentredString(width / 2, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = SOURCE.read_text(encoding="utf-8")
    story = parse_markdown(md)

    doc = SimpleDocTemplate(
        str(OUT_FILE),
        pagesize=letter,
        rightMargin=0.52 * inch,
        leftMargin=0.52 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.62 * inch,
        title="GUDS-EDL Theory Changes",
        author="Codex",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
