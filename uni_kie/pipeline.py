from pathlib import Path
from typing import List, Union

from uni_kie.constants import MODELS, PARSERS, PROMPT_VARIANTS, TOKENIZERS
from uni_kie.models.baseline import BaselineModel
from uni_kie.models.model import AbstractModel, LargeLanguageModel
from uni_kie.parser import Parser
from uni_kie.pdf_to_text.pdf_to_text import AbstractPDFToTextModel


class AbstractPipeline:
    def __init__(self, model: AbstractModel, pdf_to_text_model: AbstractPDFToTextModel):
        self.model = model
        self.pdf_to_text_model = pdf_to_text_model

    def predict(self, file_path: Union[str, Path], gold_keys: List) -> dict:
        raise NotImplementedError

    def predict_directory(self, directory_path: Union[str, Path]) -> List[dict]:
        raise NotImplementedError


class LLMPipeline(AbstractPipeline):
    """
    Consists of OCR_MODEL, PROMPT_VARIANT, MODEL, PARSER.
    # 1. takes in some file
    # 2. runs OCR_MODELS on it
    # 3. creates a prompt with the OCR_MODELS'd text
    # 4. prompts the model with the prompt
    # 5. parses the model's output
    """

    def __init__(
        self,
        model: LargeLanguageModel,
        pdf_to_text_model: AbstractPDFToTextModel,
        prompt_variant: PROMPT_VARIANTS,
        parser: Parser,
    ):
        """
        Initializes the pipeline.

        :param model: The model to use for inference.
        :param pdf_to_text_model: The OCR model to use for inference.
        :param prompt_variant: The prompt variant to use for inference.
        :param parser: The parser to use for inference.

        :type model: MODELS
        :type pdf_to_text_model: AbstractPDFToTextModel
        :type prompt_variant: PROMPT_VARIANTS
        :type parser: Parser
        """
        super().__init__(model, pdf_to_text_model)
        self.prompt_variant = prompt_variant
        self.parser = parser

    def __repr__(self):
        return f"LLMPipeline(pdf_to_text_model={self.pdf_to_text_model}, prompt_variant={self.prompt_variant}, model={self.model}, parser={self.parser})"

    @staticmethod
    def _key_list_to_string(key_list: List):
        return ", ".join([f'"{key}"' for key in key_list])

    def _generate_prompt(self, ocr_text: str, prompt_keys: List) -> str:
        if self.prompt_variant == PROMPT_VARIANTS.NEUTRAL:
            prompt = f'{ocr_text}\n\nExtract {self._key_list_to_string(prompt_keys)} from the document above. If you can\'t find a key-value pair in the document set the value to "null".\n\nKey: Value\n{prompt_keys[0]}:'

        tokenized_prompt = TOKENIZERS.GPT2_TOKENIZER_FAST(prompt)
        prompt_number_of_tokens = len(tokenized_prompt["input_ids"])

        if prompt_number_of_tokens > self.model.max_input_tokens:
            # subtracting 5 to make sure that the prompt is not too long
            truncated_tokenized_prompt = (
                tokenized_prompt["input_ids"][: self.model.max_input_tokens // 2 - 5]
                + tokenized_prompt["input_ids"][-self.model.max_input_tokens // 2 - 5 :]
            )
            prompt = TOKENIZERS.GPT2_TOKENIZER_FAST.decode(truncated_tokenized_prompt)

        return prompt

    def _parse_model_output(self, model_output: str, gold_keys) -> Union[dict, str]:
        return self.parser.parse_single_model_output(model_output, gold_keys)

    def predict(
        self, file_path: Union[str, Path], prompt_keys: List[str]
    ) -> Union[dict, str]:
        text = self.pdf_to_text_model.get_text(file_path)
        prompt = self._generate_prompt(text, prompt_keys)
        model_output = self.model.predict(prompt)
        parsed_output = self._parse_model_output(model_output, prompt_keys)
        return parsed_output

    def predict_directory(self, directory_path: Union[str, Path]):
        pass


class BaselinePipeline(AbstractPipeline):
    def __init__(self, pdf_to_text_model: AbstractPDFToTextModel):
        super().__init__(model=BaselineModel(), pdf_to_text_model=pdf_to_text_model)

    def __repr__(self):
        return f"BaselinePipeline(pdf_to_text_model={self.pdf_to_text_model}, model={self.model})"

    def predict(self, file_path: Union[str, Path]):
        text = self.pdf_to_text_model.get_text(file_path)
        prediction = self.model.predict(text)
        return prediction

    def predict_directory(self, directory_path: Union[str, Path]):
        pass
