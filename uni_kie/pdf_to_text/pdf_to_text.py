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
        """
        Extracts text from a PDF file using PyMuPDF.

        It automatically detects whether the PDF is searchable or not and extracts text accordingly. Essentially
        if it's not searchable it runs TesseractOCR on the PDF and returns the text.

        This includes the optimal handling of e.g. a PDF that is generally searchable but has a few pages that are
        not searchable or has images with text that are not searchable.

        :param file_path:
        :return:
        """
        with PyMuPDF.open(file_path, filetype="pdf") as doc:
            text = ""
            for page in doc:
                partial_tp = page.get_textpage_ocr(flags=0, full=False)
                text += page.get_text(textpage=partial_tp, sort=True) + "\n"
        return text
