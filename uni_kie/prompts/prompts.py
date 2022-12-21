from typing import List, Optional

from transformers import GPT2TokenizerFast

STOP_KEY = "\n<|stop key|>"


class Prompt:
    def __init__(self, prompt_keys: List[str], shots: Optional[List[dict]] = None):
        self.prompt_keys = prompt_keys

    def __repr__(self):
        return self.__class__.__name__

    def get_model_input(input_doc: str) -> str:
        """
        Essentially adds the prompt (e.g. "Extract Invoice number from the document below: {input_doc}") to a given document and also adds the shots (if any).
        """
        raise NotImplementedError

    def _key_list_to_string(self, key_list: List) -> str:
        return ", ".join([f'"{key}"' for key in key_list])


class NeutralPrompt(Prompt):
    def __init__(self, prompt_keys: List[str], shots: Optional[List[dict]] = None):
        super().__init__(prompt_keys=prompt_keys)
        self.prompt_text = f'\n\nExtract {self._key_list_to_string(self.prompt_keys + [STOP_KEY[1:]])} from the document above. If you can\'t find a key-value pair in the document set the value to "null".\n\nKey: Value\n{self.prompt_keys[0]}:'
        self.tokenized_prompt_text = GPT2TokenizerFast.from_pretrained("gpt2")(
            self.prompt_text
        )  # TODO: use constants.py (can't because circular import)
        self.prompt_number_of_tokens = len(self.tokenized_prompt_text["input_ids"])
        self.prompt_char_length = len(self.prompt_text)
        self.start_of_document = ""
        self.end_of_document = ""

        self.shots = False
        self.model_input_shots_number_of_tokens = 0

        if shots:
            self.shots = True
            self.model_input_shots = ""
            for shot in shots:
                self.model_input_shots += f"{self.start_of_document}{shot['input']}{self.end_of_document}{self.prompt_text}{shot['target_model_output']}\n\n"

            self.tokenized_model_input_shots = GPT2TokenizerFast.from_pretrained(
                "gpt2"
            )(self.model_input_shots)
            self.model_input_shots_number_of_tokens = len(
                self.tokenized_model_input_shots["input_ids"]
            )

            self.start_of_document = ""
            self.end_of_document = ""

        self.tokenized_start_of_document = GPT2TokenizerFast.from_pretrained("gpt2")(
            self.start_of_document
        )
        self.start_of_document_number_of_tokens = len(
            self.tokenized_start_of_document["input_ids"]
        )
        self.tokenized_end_of_document = GPT2TokenizerFast.from_pretrained("gpt2")(
            self.end_of_document
        )
        self.end_of_document_number_of_tokens = len(
            self.tokenized_end_of_document["input_ids"]
        )

    def __repr__(self):
        return super().__repr__()

    def get_model_input(self, input_doc: str) -> str:
        if self.shots:
            return f"{self.model_input_shots}{self.start_of_document}{input_doc}{self.end_of_document}{self.prompt_text}"

        else:
            return f"{self.start_of_document}{input_doc}{self.end_of_document}{self.prompt_text}"
