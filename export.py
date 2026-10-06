"""
Export utility for generating Markdown and PDF reports.
"""

import re
from fpdf import FPDF


class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Google Photos AI Research Assistant - Review Analysis Report", border=0, ln=1, align="R")
        self.line(10, 18, 200, 18)
        self.ln(4)
        self.set_x(10)  # Reset X position to left margin

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def sanitize_latin1(text: str) -> str:
    """
    Sanitize text to Latin-1 compatible string for FPDF standard fonts.
    Replaces non-latin1 Unicode symbols with standard ASCII equivalents.
    """
    replacements = {
        "•": "-",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "⭐": "*",
        "👍": "[+]",
        "👎": "[-]",
        "🔍": "[Search]",
        "☁️": "[Cloud]",
        "🔄": "[Sync]",
        "🎨": "[UI]",
        "📱": "[App]",
        "🚀": "[Launch]",
        "✨": "[Report]"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", "replace").decode("latin-1")


def convert_markdown_to_pdf(markdown_text: str) -> bytes:
    """
    Convert markdown formatted report string into PDF bytes using fpdf2.
    """
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.set_margins(10, 20, 10)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_x(10)

    lines = markdown_text.split("\n")
    for line in lines:
        line_clean = sanitize_latin1(line.strip())
        if not line_clean:
            pdf.ln(3)
            pdf.set_x(10)
            continue

        # Handle horizontal divider lines
        if re.match(r"^[\-\=\*]{3,}$", line_clean):
            pdf.ln(2)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
            pdf.set_x(10)
            continue

        # Skip table header separator lines like | :--- | :--- |
        if re.match(r"^\|[\s\:\-\|]+\|$", line_clean):
            continue

        # Format markdown tables as clean text
        if line_clean.startswith("|") and line_clean.endswith("|"):
            cells = [c.strip() for c in line_clean.strip("|").split("|")]
            line_clean = "  |  ".join([c for c in cells if c])

        if line_clean.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(33, 37, 41)
            pdf.multi_cell(0, 8, line_clean.replace("# ", "").strip())
            pdf.ln(2)
        elif line_clean.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(13, 110, 253)
            pdf.multi_cell(0, 7, line_clean.replace("## ", "").strip())
            pdf.ln(2)
        elif line_clean.startswith("### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, line_clean.replace("### ", "").strip())
            pdf.ln(1)
        elif line_clean.startswith("- ") or line_clean.startswith("* "):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            text = line_clean[2:].strip()
            text = re.sub(r"[\*\_]", "", text)
            pdf.multi_cell(0, 5, f"  -  {text}")
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            text = re.sub(r"[\*\_]", "", line_clean)
            pdf.multi_cell(0, 5, text)

        pdf.set_x(10)  # Always ensure X returns to left margin

    return bytes(pdf.output())
