import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Preformatted, HRFlowable, ListFlowable, ListItem,
)

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", FONTS / "arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", FONTS / "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", FONTS / "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FONTS / "consola.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")
GREY = colors.HexColor("#666666")
LIGHT = colors.HexColor("#f2f5f8")
BORDER = colors.HexColor("#c9d3dd")

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontName="Body",
                      fontSize=10, leading=15, textColor=INK, alignment=TA_LEFT,
                      spaceAfter=6)
h1 = ParagraphStyle("h1", parent=body, fontName="Body-Bold", fontSize=20,
                    leading=25, textColor=ACCENT, spaceBefore=6, spaceAfter=10)
h2 = ParagraphStyle("h2", parent=body, fontName="Body-Bold", fontSize=15,
                    leading=19, textColor=ACCENT, spaceBefore=14, spaceAfter=6)
h3 = ParagraphStyle("h3", parent=body, fontName="Body-Bold", fontSize=12,
                    leading=16, textColor=INK, spaceBefore=10, spaceAfter=4)
h4 = ParagraphStyle("h4", parent=body, fontName="Body-Bold", fontSize=10.5,
                    leading=14, textColor=ACCENT, spaceBefore=8, spaceAfter=3)
quote = ParagraphStyle("quote", parent=body, leftIndent=12, textColor=GREY,
                       fontName="Body-Italic", borderPadding=(0, 0, 0, 6))
cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11, spaceAfter=0)
cell_h = ParagraphStyle("cell_h", parent=cell, fontName="Body-Bold",
                        textColor=colors.white)
code = ParagraphStyle("code", parent=styles["Code"], fontName="Mono",
                      fontSize=8.5, leading=11.5, textColor=INK)


def inline(text):
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Mono" size="9">\1</font>', text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", text)
    return text


def parse(md):
    lines = md.split("\n")
    flow = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            flow.append(Spacer(1, 2))
            block = Preformatted("\n".join(buf), code)
            tbl = Table([[block]], colWidths=[16 * cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            flow.append(tbl)
            flow.append(Spacer(1, 6))
            continue

        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            header = [c.strip() for c in rows[0].strip("|").split("|")]
            data_rows = rows[2:]
            table_data = [[Paragraph(inline(c), cell_h) for c in header]]
            for r in data_rows:
                cols = [c.strip() for c in r.strip("|").split("|")]
                table_data.append([Paragraph(inline(c), cell) for c in cols])
            ncol = len(header)
            tbl = Table(table_data, colWidths=[16.0 / ncol * cm] * ncol, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            flow.append(Spacer(1, 4))
            flow.append(tbl)
            flow.append(Spacer(1, 8))
            continue

        if re.match(r"^\s*([-*_])\s*\1\s*\1[\s\1]*$", line) or stripped == "---":
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=BORDER))
            flow.append(Spacer(1, 4))
            i += 1
            continue

        if stripped.startswith("#### "):
            flow.append(Paragraph(inline(stripped[5:]), h4)); i += 1; continue
        if stripped.startswith("### "):
            flow.append(Paragraph(inline(stripped[4:]), h3)); i += 1; continue
        if stripped.startswith("## "):
            flow.append(Paragraph(inline(stripped[3:]), h2)); i += 1; continue
        if stripped.startswith("# "):
            flow.append(Paragraph(inline(stripped[2:]), h1)); i += 1; continue

        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            para = Paragraph(inline(" ".join(buf)), quote)
            tbl = Table([[para]], colWidths=[16 * cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8e6")),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor("#e0a800")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            flow.append(Spacer(1, 2)); flow.append(tbl); flow.append(Spacer(1, 6))
            continue

        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(ListItem(Paragraph(inline(re.sub(r"^\s*[-*]\s+", "", lines[i])), body),
                                      leftIndent=14, value="bullet"))
                i += 1
            flow.append(ListFlowable(items, bulletType="bullet", start="•",
                                     leftIndent=10, bulletFontName="Body"))
            flow.append(Spacer(1, 4))
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(ListItem(Paragraph(inline(re.sub(r"^\s*\d+\.\s+", "", lines[i])), body),
                                      leftIndent=14))
                i += 1
            flow.append(ListFlowable(items, bulletType="1", leftIndent=10,
                                     bulletFontName="Body"))
            flow.append(Spacer(1, 4))
            continue

        if stripped == "":
            i += 1
            continue

        buf = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(
            r"^\s*(#|\||```|>|[-*]\s|\d+\.\s|---)", lines[i]
        ):
            buf.append(lines[i].strip())
            i += 1
        flow.append(Paragraph(inline(" ".join(buf)), body))

    return flow


def build(src, dst):
    md = Path(src).read_text(encoding="utf-8")
    doc = SimpleDocTemplate(dst, pagesize=letter,
                            leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Informe Tecnico Lab 3 - ASL")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Body", 8)
        canvas.setFillColor(GREY)
        canvas.drawString(2.5 * cm, 1.2 * cm, "Laboratorio 3 - Clasificacion ASL - SignBridge")
        canvas.drawRightString(letter[0] - 2.5 * cm, 1.2 * cm, "Pag. %d" % d.page)
        canvas.setStrokeColor(BORDER)
        canvas.line(2.5 * cm, 1.5 * cm, letter[0] - 2.5 * cm, 1.5 * cm)
        canvas.restoreState()

    doc.build(parse(md), onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
    print("PDF generado:", sys.argv[2])
