import pathlib
from typing import Union

from uni_kie.constants import MODELS, OCR_MODELS, PARSERS, PROMPT_VARIANTS


class Pipeline:
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
        ocr_model: Union[OCR_MODELS],
        prompt_variant: Union[PROMPT_VARIANTS],
        model: Union[MODELS],
        parser: Union[PARSERS],
    ):
        self.ocr_model = ocr_model
        self.prompt_variant = prompt_variant
        self.model = model
        self.parser = parser

    def __repr__(self):
        return f"Pipeline(ocr_model={self.ocr_model}, prompt_variant={self.prompt_variant}, model={self.model}, parser={self.parser})"

    def _run_ocr(self, file_path: Union[str, pathlib.Path]) -> str:
        return "foo"

    def _generate_prompt(self, ocr_text: str) -> str:
        return "foz"

    def _prompt_model(self, prompt: str) -> str:
        return "bar"

    def _parse_model_output(self, model_output: str):
        return "baz"

    def predict_file(self, file_path: Union[str, pathlib.Path]):
        ocr_text = self._run_ocr(file_path)
        prompt = self._generate_prompt(ocr_text)
        model_output = self._prompt_model(prompt)
        parsed_output = self._parse_model_output(model_output)
        return parsed_output

    def predict_directory(self, directory_path):
        pass
