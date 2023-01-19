from datetime import datetime

from constants import MODELS, NER_TAGGERS, PARSERS, PDF_TO_TEXT_MODELS, PROMPT_VARIANTS
from pipeline import BaselinePipeline, LLMPipeline

from uni_kie import __version__, create_logger
from uni_kie.constants import LONG_DOCUMENT_HANDLING_VARIANTS
from uni_kie.kleister_charity_constants import (
    KLEISTER_CHARITY_CONSTANTS,
    PATH_KLEISTER_CHARITY,
)
from uni_kie.models.baseline import BaselineModel, KleisterCharitySpecificBaselineModel
from uni_kie.sroie_constants import PATH_SROIE, SROIE_CONSTANTS

logger = create_logger(__name__)

if __name__ == "__main__":
    print(__version__)
    logger.info(f"Using version {__version__}")
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # KLEISTER_CHARITY
    pipeline = LLMPipeline(
        keys=KLEISTER_CHARITY_CONSTANTS.prompt_keys,
        shots=None,  # KLEISTER_CHARITY_CONSTANTS.SHOTS,  # KLEISTER_CHARITY_CONSTANTS.SHOTS or None
        model=MODELS.FLAN_T5(),
        pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER(split="dev-0"),
        prompt_variant=PROMPT_VARIANTS.NEUTRAL,
        long_document_handling_variant=LONG_DOCUMENT_HANDLING_VARIANTS.SPLIT_TO_SUBDOCUMENTS,
        parser=PARSERS.KLEISTER_CHARITY_PARSER(),
    )

    path = (
        PATH_KLEISTER_CHARITY
        / pipeline.pdf_to_text_model.split
        / "predictions"
        / f"{now}_{pipeline}.tsv"
    )

    # SROIE
    # pipeline = LLMPipeline(
    #     keys=SROIE_CONSTANTS.prompt_keys,
    #     shots=SROIE_CONSTANTS.SHOTS,  # SROIE_CONSTANTS.SHOTS or None
    #     model=MODELS.FLAN_T5(),
    #     pdf_to_text_model=PDF_TO_TEXT_MODELS.SROIE_WRAPPER(split="test"),
    #     prompt_variant=PROMPT_VARIANTS.NEUTRAL,
    #     long_document_handling_variant=LONG_DOCUMENT_HANDLING_VARIANTS.SPLIT_TO_SUBDOCUMENTS,
    #     parser=PARSERS.DICT_PARSER(),
    # )

    # path = (
    #     PATH_SROIE
    #     / pipeline.pdf_to_text_model.split
    #     / "predictions"
    #     / f"{now}_{pipeline}.tsv"
    # )

    logger.info(
        f"Using pipeline {pipeline} on {pipeline.pdf_to_text_model.split} split",
    )

    logger.info(
        f"Searching for keys: {pipeline.keys}",
    )

    with open(path, "w") as f:
        for i in range(len(pipeline.pdf_to_text_model.data)):
            logger.info(f"Predicting document {i}...")
            # KLEISTER_CHARITY
            prediction = pipeline.predict(
                pipeline.pdf_to_text_model.data.iloc[i]["filename"]
            )

            # SROIE (data is just a list of strings (the filenames))
            # prediction = pipeline.predict(
            #     pipeline.pdf_to_text_model.data[i]
            # )
            logger.info(f"Final prediction for document {i}: {prediction}")
            f.write(f"{prediction}\n")
            print(f"Progress: {i+1}/{len(pipeline.pdf_to_text_model.data)}")
    logger.info("================== DONE ==================")
    print("======================== DONE ============================")
