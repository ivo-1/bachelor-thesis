from constants import MODELS, NER_TAGGERS, PARSERS, PDF_TO_TEXT_MODELS, PROMPT_VARIANTS
from pipeline import BaselinePipeline, LLMPipeline

from uni_kie import __version__
from uni_kie.kleister_charity_constants import KLEISTER_CHARITY_CONSTANTS

if __name__ == "__main__":
    print(__version__)
    llm_pipeline = LLMPipeline(
        model=MODELS.GPT.Davinci(),
        pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER(),
        prompt_variant=PROMPT_VARIANTS.NEUTRAL,
        parser=PARSERS.KLEISTER_CHARITY_PARSER(),
    )

    baseline_pipeline = BaselinePipeline(
        pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER(),
        parser=PARSERS.KLEISTER_CHARITY_PARSER(),
        ner_tagger=NER_TAGGERS.SPACY_WEB_SM,
    )

    # KLEISTER CHARITY EXAMPLE
    prompt_keys = KLEISTER_CHARITY_CONSTANTS.prompt_keys

    # predict on the first 3 rows
    for i in range(3):
        print(
            f"Predicted key-value pairs: {baseline_pipeline.predict(baseline_pipeline.pdf_to_text_model.data.iloc[i]['filename'], prompt_keys)}"
        )

    # # OWN INVOICE EXAMPLE
    # gold_keys = ['Invoice Number', 'Total', 'Capital of Cyprus']
    # print(f"Gold keys: {gold_keys}")
    # print(
    #     f"{gold_keys[0]}:{llm_pipeline.predict('/Users/ivo/PycharmProjects/Levity/unimodal-kie/uni_kie/datasets/own_sample_invoice.pdf', gold_keys)}"
    # )
    print("======================== DONE ============================")
