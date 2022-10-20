from pathlib import Path

import pdf2image

from uni_kie.pdf_to_text.pdf_to_text import AbstractPDFToTextModel


class AbstractOCRModel(AbstractPDFToTextModel):
    def __init__(self):
        super().__init__()

    def _convert_pdf_to_images(self, file_path: Path):
        return pdf2image.convert_from_path(file_path, dpi=300)

    def get_text(self, file_path: Path) -> str:
        return self.ocr(file_path)

    def ocr(self, file_path: Path) -> str:
        raise NotImplementedError
