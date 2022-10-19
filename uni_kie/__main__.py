from pathlib import Path

from constants import MODELS, PARSERS, PROMPT_VARIANTS
from ocr.ocr import Tesseract
from pipeline import LLM_Pipeline

from uni_kie import __version__
from uni_kie.models.baseline import BaselineModel
from uni_kie.models.t0pp import T0pp

if __name__ == "__main__":
    print(__version__)
    print("Initializing Pipeline...")
    pipeline = LLM_Pipeline(
        ocr_model=Tesseract(),
        prompt_variant=PROMPT_VARIANTS.SIMPLE,
        model=T0pp(),
        parser=PARSERS.SIMPLE,
    )
    print(pipeline)
    print("Running Pipeline...")
    print(pipeline.predict("datasets/own_sample_invoice.pdf"))
    print("Done!")
