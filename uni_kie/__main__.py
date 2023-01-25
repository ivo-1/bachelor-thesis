import json
import shutil
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
    # pipeline = LLMPipeline(
    #     keys=KLEISTER_CHARITY_CONSTANTS.prompt_keys,
    #     shots=KLEISTER_CHARITY_CONSTANTS.SHOTS, # or None
    #     model=MODELS.GPT.Davinci(),
    #     pdf_to_text_model=PDF_TO_TEXT_MODELS.KLEISTER_CHARITY_WRAPPER(split="test-A"),
    #     prompt_variant=PROMPT_VARIANTS.NEUTRAL,
    #     long_document_handling_variant=LONG_DOCUMENT_HANDLING_VARIANTS.SPLIT_TO_SUBDOCUMENTS,
    #     parser=PARSERS.KLEISTER_CHARITY_PARSER(),
    # )

    # path = (
    #     PATH_KLEISTER_CHARITY
    #     / pipeline.pdf_to_text_model.split
    #     / "predictions"
    #     / f"{now}_{pipeline}.tsv"
    # )

    # SROIE
    pipeline = LLMPipeline(
        keys=SROIE_CONSTANTS.prompt_keys,
        # zero-shot
        # shots=None,
        # one-shot
        shots=SROIE_CONSTANTS.SHOTS[0:1],
        # two-shot
        # shots=SROIE_CONSTANTS.SHOTS[0:2],
        model=MODELS.GPT.NeoX(),
        pdf_to_text_model=PDF_TO_TEXT_MODELS.SROIE_WRAPPER(split="test"),
        prompt_variant=PROMPT_VARIANTS.NEUTRAL,
        long_document_handling_variant=LONG_DOCUMENT_HANDLING_VARIANTS.SPLIT_TO_SUBDOCUMENTS,
        parser=PARSERS.DICT_PARSER(),
    )

    # BASELINE
    # pipeline = BaselinePipeline(
    #     keys=SROIE_CONSTANTS.prompt_keys,
    #     model=BaselineModel,
    #     pdf_to_text_model=PDF_TO_TEXT_MODELS.SROIE_WRAPPER(split="test"),
    #     parser=PARSERS.DICT_PARSER(),
    #     ner_tagger=NER_TAGGERS.SPACY_WEB_SM,
    #     error_percentage=0.18,
    #     allowed_entity_range=40,
    # )

    folder_path = (
        PATH_SROIE
        / pipeline.pdf_to_text_model.split
        / "predictions"
        / f"{now}_{pipeline}"
    )

    logger.info(
        f"Using pipeline {pipeline} on {pipeline.pdf_to_text_model.split} split",
    )

    logger.info(
        f"Searching for keys: {pipeline.keys}",
    )

    # KLEISTER CHARITY
    # with open(folder_path, "w") as f:
    #     for i in range(len(pipeline.pdf_to_text_model.data)):
    #         logger.info(f"Predicting document {i}...")
    #         prediction = pipeline.predict(
    #             pipeline.pdf_to_text_model.data.iloc[i]["filename"]
    #         )
    #         logger.info(f"Final prediction for document {i}: {prediction}")
    #         f.write(f"{prediction}\n")
    #         print(f"Progress: {i+1}/{len(pipeline.pdf_to_text_model.data)}")

    # SROIE
    for i in range(len(pipeline.pdf_to_text_model.data)):
        logger.info(f"Predicting document {i}...")
        prediction = pipeline.predict(
            pipeline.pdf_to_text_model.data.iloc[i]["filename"]
        )
        logger.info(f"Final prediction for document {i}: {prediction}")

        filename = pipeline.pdf_to_text_model.data.iloc[i]["filename"]
        stem = filename.stem
        stem += ".txt"
        folder_path.mkdir(parents=True, exist_ok=True)

        with open(folder_path / stem, "w") as f:
            f.write(json.dumps(prediction, indent=4))

        print(f"Progress: {i+1}/{len(pipeline.pdf_to_text_model.data)}")
    shutil.make_archive(folder_path, "zip", folder_path)

    logger.info("================== DONE ==================")
    print("======================== DONE ============================")
