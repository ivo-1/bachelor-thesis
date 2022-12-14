from pathlib import Path
from typing import List, Union

from uni_kie import create_logger
from uni_kie.constants import (
    LONG_DOCUMENT_HANDLING_VARIANTS,
    MODELS,
    NER_TAGGERS,
    PARSERS,
    PROMPT_VARIANTS,
    TOKENIZERS,
)
from uni_kie.models.baseline import AbstractBaselineModel, BaselineModel
from uni_kie.models.model import AbstractModel, LargeLanguageModel
from uni_kie.parsers.parser import Parser
from uni_kie.pdf_to_text.pdf_to_text import AbstractPDFToTextModel
from uni_kie.prompts.prompts import Prompt

logger = create_logger(__name__)


class AbstractPipeline:
    def __init__(
        self,
        keys: List[str],
        model: AbstractModel,
        pdf_to_text_model: AbstractPDFToTextModel,
        parser: Parser,
    ):
        self.keys = keys
        self.model = model
        self.pdf_to_text_model = pdf_to_text_model
        self.parser = parser

    def predict(self, file_path: Union[str, Path]) -> dict:
        raise NotImplementedError

    def predict_directory(self, directory_path: Union[str, Path]) -> List[dict]:
        raise NotImplementedError


class LLMPipeline(AbstractPipeline):
    """
    Consists of OCR_MODEL, PROMPT_VARIANT, MODEL, PARSER.
    # 1. takes in some file
    # 2. gets the text (either from OCR or from the dataset)
    # 3. creates a prompt with the OCR_MODELS'd text
    # 4. prompts the model with the prompt
    # 5. parses the model's output
    """

    def __init__(
        self,
        keys: List[str],
        model: LargeLanguageModel,
        pdf_to_text_model: AbstractPDFToTextModel,
        prompt_variant: Prompt,
        long_document_handling_variant: LONG_DOCUMENT_HANDLING_VARIANTS,
        parser: Parser,
    ):
        """
        Initializes the inference pipeline.

        :param keys: keys of interest for a given document (used verbatim in the prompt)
        :param model: model to be used
        :param pdf_to_text_model: model to be used for extracting text from pdfs
        :param prompt_variant: prompt variant to be used
        :param long_document_handling_variant: how to handle long documents
        :param parser: parser to be used
        """
        super().__init__(
            keys=keys, model=model, pdf_to_text_model=pdf_to_text_model, parser=parser
        )
        self.prompt_variant = prompt_variant(prompt_keys=keys)
        self.long_document_handling_variant = long_document_handling_variant

    def __repr__(self):
        return f"LLMPipeline(prompt_variant={self.prompt_variant}, model={self.model}, parser={self.parser})"

    def get_parsed_output(self, model_output: List[str], prompt_keys: List[str]) -> str:
        if (
            len(model_output) != 1
        ):  # doc was too long and subdoc handling chosen -> we have multiple model_outputs (one per subdoc)
            parsed_model_output = [
                PARSERS.DICT_PARSER.parse_model_output(model_output, prompt_keys)
                for model_output in model_output
            ]

            unified_dict = {}
            for key in prompt_keys:
                values = [
                    model_output[key]
                    for model_output in parsed_model_output
                    if key in model_output
                ]
                if (
                    len(values) == 0
                ):  # if in no subdoc the key was found, we don't put it in the unified dict
                    logger.info(f"Key not found in any subdoc {key}")
                    continue
                else:
                    lower_values = [value.lower() for value in values]
                    logger.info(f"Unification necessary for key {key}")
                    logger.info(
                        f"Unifying {len(values)} (lowered) values {lower_values}"
                    )

                    most_common = max(
                        list(dict.fromkeys(lower_values)),
                        key=lower_values.count,  # picks the first one (according to the order of the list (-> earlier pages are preferred)) if there are multiple max values
                    )

                    unified_dict[key] = values[
                        lower_values.index(most_common)
                    ]  # pick the first one in the not lowered list (restores the original casing)
                    logger.info(f"Unified value {unified_dict[key]}")

            if isinstance(self.parser, PARSERS.DICT_PARSER):
                return unified_dict
            elif isinstance(self.parser, PARSERS.KLEISTER_CHARITY_PARSER):
                return self.parser._dict_to_kleister_charity(unified_dict, prompt_keys)

        else:
            logger.info("No subdocs necessary")
            return self.parser.parse_model_output(model_output[0], prompt_keys)

    def get_model_output(self, model_input: str) -> List[str]:
        """
        Gets the model's output for a given model_input.

        Includes handling of documents that are too long for the model
        in which case multiple model outputs may be returned.
        """
        tokenized_model_input = TOKENIZERS.GPT2_TOKENIZER_FAST(model_input)
        number_of_tokens_model_input = len(tokenized_model_input["input_ids"])
        if number_of_tokens_model_input > self.model.max_input_tokens:
            logger.info(
                f"Document is too long for the model. Number of tokens: {number_of_tokens_model_input}. Max number of tokens: {self.model.max_input_tokens}."
            )
            if (
                self.long_document_handling_variant
                == LONG_DOCUMENT_HANDLING_VARIANTS.TRUNCATE_MIDDLE
            ):
                # subtracting 5 to make sure that the prompt is not too long
                truncated_tokenized_model_input = (
                    tokenized_model_input["input_ids"][
                        : self.model.max_input_tokens // 2 - 5
                    ]
                    + tokenized_model_input["input_ids"][
                        -self.model.max_input_tokens // 2 - 5 :
                    ]
                )
                truncated_model_input = TOKENIZERS.GPT2_TOKENIZER_FAST.decode(
                    truncated_tokenized_model_input
                )
                return [self.model.predict(truncated_model_input)]

            elif (
                self.long_document_handling_variant
                == LONG_DOCUMENT_HANDLING_VARIANTS.SPLIT_TO_SUBDOCUMENTS
            ):
                prompt_length = self.prompt_variant.prompt_number_of_tokens
                overlap_no_tokens = (
                    20  # how many tokens to overlap between subdocuments
                )

                # cut off the prompt from the model input
                model_input_without_prompt_input_ids = tokenized_model_input[
                    "input_ids"
                ][:-prompt_length]
                subdocuments = []

                for i in range(
                    0,
                    number_of_tokens_model_input,
                    self.model.max_input_tokens
                    - overlap_no_tokens
                    - prompt_length
                    - 5,  # subtracting 5 to make sure that the prompt is not too long
                ):
                    subdocument = model_input_without_prompt_input_ids[
                        i : i
                        + self.model.max_input_tokens
                        - prompt_length
                        - 5  # subtracting 5 to make sure that the prompt is not too long
                    ]
                    subdocuments.append(subdocument)

                subdocuments = [
                    TOKENIZERS.GPT2_TOKENIZER_FAST.decode(subdocument)
                    for subdocument in subdocuments
                ]
                subdocuments_with_prompt = [
                    self.prompt_variant.get_model_input(subdocument)
                    for subdocument in subdocuments
                ]
                logger.info(
                    f"Split document into {len(subdocuments_with_prompt)} subdocuments."
                )
                subdocuments_predictions = [
                    self.model.predict(subdocument)
                    for subdocument in subdocuments_with_prompt
                ]
                return subdocuments_predictions

        else:
            prediction = self.model.predict(model_input)
            logger.info(f"Raw prediction for document: {prediction}")
            return [prediction]

    def predict(self, file_path: Union[str, Path]) -> Union[dict, str]:
        """
        Return type depends on the parser.

        DictParser -> dict
        KleisterCharityParser -> str
        """
        text = self.pdf_to_text_model.get_text(file_path)
        model_input = self.prompt_variant.get_model_input(text)
        model_output = self.get_model_output(model_input)
        parsed_output = self.get_parsed_output(model_output, self.keys)
        return parsed_output

    def predict_directory(
        self, directory_path: Union[str, Path]
    ) -> Union[List[dict], List[str]]:
        pass


class BaselinePipeline(AbstractPipeline):
    def __init__(
        self,
        keys: List[str],
        model: AbstractBaselineModel,
        pdf_to_text_model: AbstractPDFToTextModel,
        parser: Parser,
        ner_tagger: NER_TAGGERS,
        error_percentage: float,
        allowed_entity_range: int,
    ):
        super().__init__(
            keys=keys,
            model=model(
                ner_tagger=ner_tagger,
                error_percentage=error_percentage,
                allowed_entity_range=allowed_entity_range,
            ),
            pdf_to_text_model=pdf_to_text_model,
            parser=parser,
        )

    def __repr__(self):
        return f"BaselinePipeline(pdf_to_text_model={self.pdf_to_text_model}, model={self.model}, parser={self.parser}, ner_tagger={self.model.ner_tagger})"

    def predict(self, file_path: Union[str, Path]) -> Union[dict, str]:
        text = self.pdf_to_text_model.get_text(file_path)
        model_output = self.model.predict(text, self.keys)
        parsed_output = self.parser.parse_model_output(model_output, self.keys)
        return parsed_output

    def predict_directory(
        self, directory_path: Union[str, Path]
    ) -> Union[List[dict], List[str]]:
        pass
