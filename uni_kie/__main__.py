from constants import MODELS, PARSERS, PDF_TO_TEXT_MODELS, PROMPT_VARIANTS
from pipeline import BaselinePipeline, LLMPipeline

from uni_kie import __version__
from uni_kie.pdf_to_text.pdf_to_text import KleisterCharityWrapper

if __name__ == "__main__":
    print(__version__)
    llm_pipeline = LLMPipeline(
        model=MODELS.GPT.Davinci,
        pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER,
        prompt_variant=PROMPT_VARIANTS.NEUTRAL,
        parser=PARSERS.JSON_PARSER,
    )
    # print(llm_pipeline)
    # print("Running LLMPipeline...")
    #
    # # OWN INVOICE EXAMPLE
    # gold_keys = ['Invoice Number', 'Total', 'Capital of Cyprus']
    # print(f"Gold keys: {gold_keys}")
    # print(
    #     f"{gold_keys[0]}:{llm_pipeline.predict('/Users/ivo/PycharmProjects/Levity/unimodal-kie/uni_kie/datasets/own_sample_invoice.pdf', gold_keys)}"
    # )

    # KLEISTER CHARITY EXAMPLE
    prompt_keys = list(KleisterCharityWrapper().gold_key_to_prompt_key.values())

    with open("datasets/kleister_charity_test_set/in-for-testing.tsv", "r") as f:
        for line in f:
            line = line.split("\t")
            print(f"Parsed model output:\n{llm_pipeline.predict(line[0], prompt_keys)}")

    print("======================== DONE ============================")
