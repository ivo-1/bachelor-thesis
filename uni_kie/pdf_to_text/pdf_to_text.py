from pathlib import Path

import fitz as PyMuPDF
import pandas as pd


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
        Extracts text from a PDF file using PY_MU_PDF.

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


class KleisterCharityWrapper(AbstractPDFToTextModel):
    """
    A wrapper of the Kleister Charity dataset which uses the text
    that is already provided with the dataset.
    """

    def __init__(self):
        super().__init__()
        self.gold_keys = [
            "address__post_town",
            "address__postcode",
            "address__street_line",
            "charity_name",
            "charity_number",
            "income_annually_in_british_pounds",
            "report_date",
            "spending_annually_in_british_pounds",
        ]
        self.prompt_key_to_gold_key = {
            "Address (post town)": "address__post_town",
            "Address (post code)": "address__post_code",
            "Address (street)": "address__street_line",
            "Charity Name": "charity_name",
            "Charity Number": "charity_number",
            "Annual Income": "income_annually_in_british_pounds",
            "Report Date (YYYY-MM-DD, ISO8601)": "report_date",
            "Annual Spending": "spending_annually_in_british_pounds",
        }
        self.gold_key_to_prompt_key = {
            "address__post_town": "Address (post town)",
            "address__postcode": "Address (post code)",
            "address__street_line": "Address (street)",
            "charity_name": "Charity Name",
            "charity_number": "Charity Number",
            "income_annually_in_british_pounds": "Annual Income",
            "report_date": "Report Date (YYYY-MM-DD, ISO8601)",
            "spending_annually_in_british_pounds": "Annual Spending",
        }
        self.data = self._load_data()

    def __repr__(self):
        return f"KleisterCharityWrapper(AbstractPDFToTextModel)"

    @staticmethod
    def _load_data() -> pd.DataFrame:
        path_to_data = "./datasets/kleister_charity_test_set/in-for-testing.tsv"
        path_to_headers = "./datasets/kleister_charity_test_set/in-header.tsv"

        data = pd.read_csv(path_to_data, sep="\t", header=None)
        headers = pd.read_csv(path_to_headers, sep="\t", header=None)
        data.columns = headers.iloc[0]
        data = data.drop(columns=["text_djvu", "text_tesseract", "text_textract"])
        return data

    def get_text(self, file_path: Path) -> str:
        pass
