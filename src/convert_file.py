from markitdown import MarkItDown
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os

def convert_file(url):
    try:
        print("[🔍] Trying structured text extraction via MarkItDown...")
        md = MarkItDown(enable_plugins=True)
        result = md.convert(url)

        if result.text_content.strip():
            print("[✔] Markdown extracted via MarkItDown, Please wait...")
            return result.text_content
        else:
            print("[⚠️] No text found via MarkItDown, falling back to OCR...")
    except Exception as e:
        print(f"[❌] MarkItDown failed: {e}")
        print("[🔁] Falling back to OCR...")

    try:
        ext = os.path.splitext(url)[-1].lower()
        images = []

        if ext == '.pdf':
            images = convert_from_path(url)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            images = [Image.open(url)]
        else:
            raise Exception(f"Unsupported file format: {ext}")

        full_text = ""
        for img in images:
            text = pytesseract.image_to_string(img, lang='eng')
            full_text += "\n" + text.strip()

        if not full_text.strip():
            raise Exception("OCR returned no text.")

        print("[✔] Text extracted via OCR.")
        return full_text
    except Exception as e:
        print(f"[❌] OCR failed: {e}")
        raise
