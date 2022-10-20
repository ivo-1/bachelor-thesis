from pathlib import Path

import pytesseract

from uni_kie.pdf_to_text.ocr.ocr import AbstractOCRModel


class Tesseract(AbstractOCRModel):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"Tesseract()"

    def ocr(self, file_path: Path) -> str:
        img = self._convert_pdf_to_image(file_path)
        return pytesseract.image_to_string(img)
