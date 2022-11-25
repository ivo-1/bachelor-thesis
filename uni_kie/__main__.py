from datetime import datetime

from constants import MODELS, NER_TAGGERS, PARSERS, PDF_TO_TEXT_MODELS, PROMPT_VARIANTS
from pipeline import BaselinePipeline, LLMPipeline

from uni_kie import __version__, create_logger
from uni_kie.constants import LONG_DOCUMENT_HANDLING_VARIANTS
from uni_kie.kleister_charity_constants import (
    KLEISTER_CHARITY_CONSTANTS,
    PATH_KLEISTER_CHARITY,
)

logger = create_logger(__name__)

if __name__ == "__main__":
    print(__version__)
    logger.info(f"Using version {__version__}")
    pipeline = LLMPipeline(
        keys=KLEISTER_CHARITY_CONSTANTS.prompt_keys,
        model=MODELS.GPT.NeoX(),
        pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER(split="dev-0"),
        prompt_variant=PROMPT_VARIANTS.NEUTRAL,
        long_document_handling_variant=LONG_DOCUMENT_HANDLING_VARIANTS.SPLIT_TO_SUBDOCUMENTS,
        parser=PARSERS.KLEISTER_CHARITY_PARSER(),
    )

    # pipeline = BaselinePipeline(
    #     keys=KLEISTER_CHARITY_CONSTANTS.SPECIFIC_BASELINE.prompt_keys,
    #     model=KleisterCharitySpecificBaselineModel,
    #     pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER(
    #         split="dev-0"
    #     ),
    #     parser=PARSERS.KLEISTER_CHARITY_PARSER(),
    #     ner_tagger=NER_TAGGERS.SPACY_WEB_SM,
    #     error_percentage=0.18,
    #     allowed_entity_range=40, # entity has to be within this amount of chars of a found key
    # )

    logger.info(
        f"Using pipeline {pipeline} on {pipeline.pdf_to_text_model.split} split",
    )
    logger.info(
        f"Using long_document_handling_variant: {pipeline.long_document_handling_variant}",
    )
    logger.info(
        f"Searching for keys: {pipeline.keys}",
    )

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = (
        PATH_KLEISTER_CHARITY
        / pipeline.pdf_to_text_model.split
        / "predictions"
        / f"{pipeline}_{now}.tsv"
    )
    with open(path, "w") as f:
        for i in range(len(pipeline.pdf_to_text_model.data)):
            logger.info(f"Predicting document {i}...")
            prediction = pipeline.predict(
                pipeline.pdf_to_text_model.data.iloc[i]["filename"]
            )
            logger.info(f"Final prediction for document {i}: {prediction}")
            f.write(f"{prediction}\n")
            print(f"Progress: {i+1}/{len(pipeline.pdf_to_text_model.data)}")
    logger.info("================== DONE ==================")
    print("======================== DONE ============================")
