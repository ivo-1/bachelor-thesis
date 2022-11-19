from pathlib import Path
from typing import Union

import fitz as PyMuPDF
import pandas as pd

from uni_kie.kleister_charity_constants import KLEISTER_CHARITY_CONSTANTS


class AbstractPDFToTextModel:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def get_text(self, file_path: Union[Path, str]) -> str:
        raise NotImplementedError


class PyMuPDFWrapper(AbstractPDFToTextModel):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return super().__repr__()

    def get_text(self, file_path: Union[Path, str]) -> str:
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

    def __init__(self, split: str = "dev-0"):
        super().__init__()
        self.split = split
        self.data = self._load_data()

    def __repr__(self):
        return super().__repr__()

    def _load_data(self) -> pd.DataFrame:
        print(f">>>>>>>>>>>>>> LOADING {self.split} SET")

        data = pd.read_csv(
            KLEISTER_CHARITY_CONSTANTS.split_to_path[self.split], sep="\t"
        )
        return data

    def get_text(self, file_path: Union[Path, str]) -> str:
        text = self.data.loc[
            self.data["filename"] == file_path, "text_best_cleaned"
        ].values[0]
        return text
