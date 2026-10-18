"""Reproducibly convert the project's simple Markdown structure to PDF."""
from html import escape
from pathlib import Path
import re

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "PS1_research.md"
TARGET = ROOT / "research" / "PS1_research.pdf"


def markdown_to_story(text: str):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="DocTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor="#17365D", spaceAfter=14))
    styles["Heading1"].textColor = "#17365D"
    styles["Heading2"].textColor = "#245A81"
    styles["BodyText"].leading = 12
    story = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 2)); continue
        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), styles["DocTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), styles["Heading1"]))
        elif line.startswith("### "):
            story.append(Paragraph(escape(line[4:]), styles["Heading2"]))
        elif re.match(r"\d+\. ", line):
            story.append(Paragraph(escape(line), styles["BodyText"]))
        else:
            safe = escape(line)
            safe = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", safe)
            safe = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", r"<super>[\1]</super>", safe)
            story.append(Paragraph(safe, styles["BodyText"]))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColorRGB(.35, .4, .46)
    canvas.drawString(20 * mm, 12 * mm, "IVDC Autonomy 2026 - PS1 Computer Vision")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    doc = SimpleDocTemplate(str(TARGET), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=18*mm, bottomMargin=20*mm, title="PS1 Computer Vision Research")
    doc.build(markdown_to_story(SOURCE.read_text(encoding="utf-8")), onFirstPage=footer, onLaterPages=footer)
    print(f"Created {TARGET}")
