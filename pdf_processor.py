import fitz
from PIL import Image
import io

def process_pdf(pdf_path):
    print(f"Processing PDF: {pdf_path}")
    text_content = ""
    images = []

    doc = fitz.open(pdf_path)
    for page in doc:
        text_content += page.get_text()
        
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    print(f"Extracted {len(text_content)} characters and {len(images)} images")
    return text_content, images
