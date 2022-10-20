from pathlib import Path

import fitz as PyMuPDF


class AbstractPDFToTextModel:
    def __init__(self):
        pass

    def get_text(self, file_path: Path) -> str:
        raise NotImplementedError


class PyMuPDFWrapper(AbstractPDFToTextModel):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"PyMuPDFWrapper(AbstractPDFToTextModel)"

    def get_text(self, file_path: Path) -> str:
        with PyMuPDF.open(file_path, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
        return text
