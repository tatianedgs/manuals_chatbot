from typing import List, Tuple
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm

# messages: List[(role, text)]

def export_chat_pdf(path: str, messages: List[Tuple[str, str]], logo_path: str | None = None):
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4

    def on_page_header():
        if logo_path:
            try:
                img = ImageReader(logo_path)
                c.drawImage(img, x=2*cm, y=H-3*cm, width=3*cm, height=2*cm, preserveAspectRatio=True, mask='auto')
            except Exception:
                pass
        c.setFont("Helvetica-Bold", 12)
        c.drawString(6*cm, H-2*cm, "NUPETR/IDEMA-RN — Chat de Parecer Técnico")
        c.line(2*cm, H-2.2*cm, W-2*cm, H-2.2*cm)

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=3*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Diálogo exportado</b>", styles['Title']))
    story.append(Spacer(1, 12))

    for role, text in messages:
        style = styles['Normal']
        if role == 'user':
            story.append(Paragraph(f"<b>Usuário:</b> {text}", style))
        else:
            story.append(Paragraph(f"<b>Assistente:</b> {text}", style))
        story.append(Spacer(1, 8))

    def on_page(canvas_obj, doc_obj):
        on_page_header()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)