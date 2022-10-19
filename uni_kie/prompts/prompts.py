from typing import Union

from uni_kie.constants import PROMPT_VARIANTS


class Prompts:
    @staticmethod
    def generate_prompt(document_ocr: str, variant: Union[PROMPT_VARIANTS]) -> str:
        return document_ocr
