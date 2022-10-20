from pathlib import Path

import pytesseract

from uni_kie.pdf_to_text.ocr.ocr import AbstractOCRModel


class Tesseract(AbstractOCRModel):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"Tesseract()"

    def ocr(self, file_path: Path) -> str:
        images = self._convert_pdf_to_images(file_path)
        ocr_string = ""
        for image in images:
            ocr_string += pytesseract.image_to_string(image)

        return ocr_string
