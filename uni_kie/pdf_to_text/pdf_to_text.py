from pathlib import Path
from typing import Union

import fitz as PyMuPDF
import numpy as np
import pandas as pd

from uni_kie.kleister_charity_constants import KLEISTER_CHARITY_CONSTANTS
from uni_kie.sroie_constants import (
    PATH_SROIE_TEST,
    PATH_SROIE_TEST_OCR,
    SROIE_CONSTANTS,
)


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

    def get_text(self, file_path: Union[Path, str], filetype: str) -> str:
        """
        Extracts text from a PDF file using PY_MU_PDF.

        It automatically detects whether the PDF is searchable or not and extracts text accordingly. Essentially
        if it's not searchable it runs TesseractOCR on the PDF and returns the text.

        This includes the optimal handling of e.g. a PDF that is generally searchable but has a few pages that are
        not searchable or has images with text that are not searchable.

        :param file_path:
        :return:
        """
        with PyMuPDF.open(file_path, filetype=filetype) as doc:
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


class SroieWrapper(AbstractPDFToTextModel):
    """
    A wrapper of the SROIE dataset which uses the text
    that is already provided with the dataset.
    """

    def __init__(self, split: str = "test"):
        super().__init__()
        self.split = split
        self.data = self._load_data()

    def __repr__(self):
        return super().__repr__()

    def _load_file_ocr(self, path: Path):
        line_list = []

        with open(path, "r", errors="ignore") as f:
            for line in f.read().splitlines():
                if len(line) == 0:
                    continue

                split_lines = line.split(",")
                text = ",".join(
                    split_lines[8:]
                )  # text may contain commas so we have to rejoin accordingly
                line_list.append(text)

        combined_text = "\n".join([line for line in line_list])
        path = path.parent.parent / "input" / path.stem.replace("ocr_boxes", "input")
        path = path.with_suffix(".jpg")

        dataframe = pd.DataFrame(
            data=[[path, combined_text]], columns=["filename", "text_sroie_ocr"]
        )
        return dataframe

    def _load_data(self) -> pd.DataFrame:
        print(f">>>>>>>>>>>>>> LOADING {self.split} SET")

        data_path = list(PATH_SROIE_TEST_OCR.iterdir())

        data = pd.DataFrame(columns=["filename", "text_sroie_ocr"])
        for path in data_path:
            row = self._load_file_ocr(path)  # a pandas dataframe with one row
            data = pd.concat([data, row], ignore_index=True)

        return data

    def get_text(self, file_path: Union[Path, str]) -> str:
        text = self.data.loc[
            self.data["filename"] == file_path, "text_sroie_ocr"
        ].values[0]
        return text


class SroieWrapperOwnOCR(PyMuPDFWrapper):
    """
    A wrapper of the SROIE dataset which in turn uses PyMuPDFWrapper
    to extract the text from the JPG files.
    """

    def __init__(self, split: str = "test"):
        super().__init__()
        self.split = split
        self.data = self._load_data()

    def __repr__(self):
        return super().__repr__()

    def get_text(self, file_path: Union[Path, str]) -> str:
        text = super().get_text(file_path, filetype="jpg")
        return text

    def _load_data(self) -> list:  # just a list of file names (absolute path)
        print(f">>>>>>>>>>>>>> LOADING {self.split} SET")
        data = list(PATH_SROIE_TEST.iterdir())
        return data
