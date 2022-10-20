from pathlib import Path
from typing import List, Union

from uni_kie.constants import MODELS, OCR_MODELS, PARSERS, PROMPT_VARIANTS
from uni_kie.models.baseline import BaselineModel
from uni_kie.models.model import AbstractModel, LargeLanguageModel
from uni_kie.ocr.ocr import AbstractOCRModel
from uni_kie.pdf_to_text.pdf_to_text import AbstractPDFToTextModel


class AbstractPipeline:
    def __init__(self, model: AbstractModel, pdf_to_text_model: AbstractPDFToTextModel):
        self.model = model
        self.pdf_to_text_model = pdf_to_text_model

    def predict(self, file_path: Union[str, Path]) -> dict:
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
    # 5. parses the model's output into a JSON
    # 6. returns the JSON
    """

    def __init__(
        self,
        model: LargeLanguageModel,
        pdf_to_text_model: AbstractPDFToTextModel,
        prompt_variant: PROMPT_VARIANTS,
        parser: PARSERS,
    ):
        """
        Initializes the pipeline.

        :param model: The model to use for inference.
        :param pdf_to_text_model: The OCR model to use for inference.
        :param prompt_variant: The prompt variant to use for inference.
        :param parser: The parser to use for inference.

        :type model: MODELS
        :type pdf_to_text_model: AbstractPDFToTextModel
        """
        super().__init__(model, pdf_to_text_model)
        self.prompt_variant = prompt_variant
        self.parser = parser

    def __repr__(self):
        return f"LLMPipeline(pdf_to_text_model={self.pdf_to_text_model}, prompt_variant={self.prompt_variant}, model={self.model}, parser={self.parser})"

    def _generate_prompt(self, ocr_text: str) -> str:
        return ocr_text

    def _prompt_model(self, prompt: str) -> str:
        return prompt

    def _parse_model_output(self, model_output: str):
        return model_output

    def predict(self, file_path: Union[str, Path]):
        text = self.pdf_to_text_model.get_text(file_path)
        prompt = self._generate_prompt(text)
        model_output = self._prompt_model(prompt)
        parsed_output = self._parse_model_output(model_output)
        return parsed_output

    def predict_directory(self, directory_path: Union[str, Path]):
        pass


class BaselinePipeline(AbstractPipeline):
    def __init__(self, ocr_model):
        super().__init__(model=BaselineModel(), pdf_to_text_model=ocr_model)

    def __repr__(self):
        return f"BaselinePipeline(pdf_to_text_model={self.pdf_to_text_model}, model={self.model})"

    def predict(self, ocr_text: str):
        return self.model.predict(ocr_text)

    def predict_file(self, file_path: Union[str, Path]):
        text = self.pdf_to_text_model.get_text(file_path)
        model_output = self.predict(text)
        return model_output

    def predict_directory(self, directory_path: Union[str, Path]):
        pass
