from typing import List, Union

from transformers import GPT2TokenizerFast

STOP_KEY = "<|stop key|>"


class Prompt:
    def __init__(self, prompt_keys: List[str]):
        self.prompt_keys = prompt_keys

    def get_model_input(input_doc: str) -> str:
        """
        Essentially adds the prompt (e.g. "Extract Invoice number from the document below: {input_doc}") to a given document.
        """
        raise NotImplementedError

    # def remove_prompt_from_model_input(self, model_input: str) -> str:
    #     """
    #     Removes the prompt from the model input.
    #     """
    #     raise NotImplementedError

    def _key_list_to_string(self, key_list: List) -> str:
        return ", ".join([f'"{key}"' for key in key_list])


class NeutralPrompt(Prompt):
    def __init__(self, prompt_keys: List[str]):
        super().__init__(prompt_keys=prompt_keys)
        self.prompt_text = f'\n\nExtract {self._key_list_to_string(self.prompt_keys + [STOP_KEY])} from the document above. If you can\'t find a key-value pair in the document set the value to "null".\n\nKey: Value\n{self.prompt_keys[0]}:'
        self.tokenized_prompt_text = GPT2TokenizerFast.from_pretrained("gpt2")(
            self.prompt_text
        )  # TODO: use constants.py (can't because circular import)
        self.prompt_number_of_tokens = len(self.tokenized_prompt_text["input_ids"])
        self.prompt_char_length = len(self.prompt_text)

    def get_model_input(self, input_doc: str) -> str:
        return f"{input_doc}{self.prompt_text}"
