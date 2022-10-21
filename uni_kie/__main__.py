from constants import PARSERS, PROMPT_VARIANTS
from pipeline import BaselinePipeline, LLMPipeline

from uni_kie import __version__
from uni_kie.models.gpt import GPT3_Davinci
from uni_kie.models.t0pp import T0pp
from uni_kie.pdf_to_text.ocr.tesseract import Tesseract
from uni_kie.pdf_to_text.pdf_to_text import PyMuPDFWrapper

if __name__ == "__main__":
    print(__version__)
    llm_pipeline = LLMPipeline(
        model=GPT3_Davinci(),
        pdf_to_text_model=PyMuPDFWrapper(),
        prompt_variant=PROMPT_VARIANTS.NEUTRAL,
        parser=PARSERS.SIMPLE,
    )
    print(llm_pipeline)
    print("Running LLMPipeline...")
    gold_keys = ["Invoice Number", "Total", "VAT Percentage", "Email address of seller"]
    print(f"Gold keys: {gold_keys}")
    print(
        f"{gold_keys[0]}:{llm_pipeline.predict('/Users/ivo/PycharmProjects/Levity/unimodal-kie/uni_kie/datasets/own_sample_invoice.pdf', gold_keys)}"
    )

    # print("Initializing BaselinePipeline...")
    # baseline_pipeline = BaselinePipeline(pdf_to_text_model=PyMuPDFWrapper())
    # print(baseline_pipeline)
    # print("Running BaselinePipeline...")
    # print(baseline_pipeline.predict("./datasets/own_sample_invoice.pdf"))

    print("======================== DONE ============================")
