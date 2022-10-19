from pathlib import Path
from typing import Union


class AbstractOCRModel:
    def __init__(self):
        pass

    def ocr(self, file_path: Path) -> str:
        raise NotImplementedError


class Tesseract(AbstractOCRModel):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"Tesseract()"

    def ocr(self, file_path: Path) -> str:
        return "foo"
