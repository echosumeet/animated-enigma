"""PDF rendering of a generated document.

Monospaced and faithful to the text rendering on purpose: the PDF exists so the corpus
can be fed to a real OCR/layout stack, and the text version stays the reference so
extraction accuracy is measured without an OCR engine in the loop.
"""

from __future__ import annotations

from .generate import GeneratedDoc

PAGE_W, PAGE_H = 842.0, 595.0  # A4 landscape: wide tables stay on one line
LEFT, TOP, LEADING, FONT_SIZE = 36.0, 40.0, 11.0, 7.5


def write_pdf(doc: GeneratedDoc, path: str) -> str:
    from reportlab.pdfgen import canvas

    cv = canvas.Canvas(path, pagesize=(PAGE_W, PAGE_H))
    cv.setTitle(f"{doc.shipment_id} {doc.doc_type}")
    y = PAGE_H - TOP
    cv.setFont("Courier", FONT_SIZE)
    for line in doc.text.splitlines():
        if y < 30:
            cv.showPage()
            cv.setFont("Courier", FONT_SIZE)
            y = PAGE_H - TOP
        cv.drawString(LEFT, y, line[:150])
        y -= LEADING
    cv.showPage()
    cv.save()
    return path
