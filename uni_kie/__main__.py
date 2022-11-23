from datetime import datetime

from constants import MODELS, NER_TAGGERS, PARSERS, PDF_TO_TEXT_MODELS, PROMPT_VARIANTS
from pipeline import BaselinePipeline, LLMPipeline

from uni_kie import __version__
from uni_kie.constants import LONG_DOCUMENT_HANDLING_VARIANTS
from uni_kie.kleister_charity_constants import (
    KLEISTER_CHARITY_CONSTANTS,
    PATH_KLEISTER_CHARITY,
)

if __name__ == "__main__":
    print(__version__)
    pipeline = LLMPipeline(
        keys=KLEISTER_CHARITY_CONSTANTS.prompt_keys,
        model=MODELS.GPT.NeoX(),
        pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER(split="dev-0"),
        prompt_variant=PROMPT_VARIANTS.NEUTRAL,
        long_document_handling_variant=LONG_DOCUMENT_HANDLING_VARIANTS.SPLIT_TO_SUBDOCUMENTS,
        parser=PARSERS.KLEISTER_CHARITY_PARSER(),
    )

    # pipeline = BaselinePipeline(
    #     keys=KLEISTER_CHARITY_CONSTANTS.prompt_keys,
    #     pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER(
    #         split="dev-0"
    #     ),
    #     parser=PARSERS.KLEISTER_CHARITY_PARSER(),
    #     ner_tagger=NER_TAGGERS.SPACY_WEB_LG,
    #     error_percentage=0.24,
    #     allowed_entity_range=40, # entity has to be within 40 chars of a found key
    # )
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    path = (
        PATH_KLEISTER_CHARITY
        / pipeline.pdf_to_text_model.split
        / "predictions"
        / f"{pipeline}_{now}.tsv"
    )
    with open(path, "w") as f:
        for i in range(len(pipeline.pdf_to_text_model.data)):
            prediction = pipeline.predict(
                pipeline.pdf_to_text_model.data.iloc[i]["filename"]
            )

            print(f"prediction: {prediction}")
            f.write(f"{prediction}\n")
            print(f"Progress: {i+1}/{len(pipeline.pdf_to_text_model.data)}")
    print("======================== DONE ============================")
