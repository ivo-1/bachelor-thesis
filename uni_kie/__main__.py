from pathlib import Path

from constants import MODELS, PARSERS, PROMPT_VARIANTS
from ocr.tesseract import Tesseract
from pipeline import BaselinePipeline, LLMPipeline

from uni_kie import __version__
from uni_kie.models.baseline import BaselineModel
from uni_kie.models.t0pp import T0pp

if __name__ == "__main__":
    print(__version__)
    print("Initializing LLMPipeline...")
    llm_pipeline = LLMPipeline(
        model=T0pp(),
        pdf_to_text_model=Tesseract(),
        prompt_variant=PROMPT_VARIANTS.SIMPLE,
        parser=PARSERS.SIMPLE,
    )
    print(llm_pipeline)
    print("Running LLMPipeline...")
    print(llm_pipeline.predict("./datasets/own_sample_invoice.pdf"))

    print("Initializing BaselinePipeline...")
    baseline_pipeline = BaselinePipeline(
        ocr_model=Tesseract(),
    )
    print(baseline_pipeline)
    print("Running BaselinePipeline...")
    print(baseline_pipeline.predict("./datasets/own_sample_invoice.pdf"))

    print("Done!")
