"""Utility script to convert text policy files to PDFs.

This script creates PDF versions of the sample policy text files
for use with the PDF loader.
"""

from pathlib import Path
import sys

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def text_to_pdf(text_path: Path, pdf_path: Path):
    """Convert a text file to PDF."""
    if not REPORTLAB_AVAILABLE:
        print("reportlab not available. Install with: pip install reportlab")
        print(f"Note: You can manually convert {text_path} to PDF using any PDF converter.")
        return False

    # Read text file
    with open(text_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    # Create PDF
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Split into paragraphs and add to PDF
    paragraphs = text_content.split("\n\n")
    for para in paragraphs:
        if para.strip():
            # Use normal style, but handle long lines
            p = Paragraph(para.strip().replace("\n", "<br/>"), styles["Normal"])
            story.append(p)
            story.append(Spacer(1, 12))

    doc.build(story)
    return True


def main():
    """Convert all text policy files to PDFs."""
    data_dir = Path(__file__).parent.parent / "data" / "policies"
    
    text_files = list(data_dir.glob("*.txt"))
    
    if not text_files:
        print("No text policy files found.")
        return 1

    for text_file in text_files:
        pdf_file = text_file.with_suffix(".pdf")
        print(f"Converting {text_file.name} to {pdf_file.name}...")
        
        if text_to_pdf(text_file, pdf_file):
            print(f"✓ Created {pdf_file.name}")
        else:
            print(f"✗ Failed to create {pdf_file.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
