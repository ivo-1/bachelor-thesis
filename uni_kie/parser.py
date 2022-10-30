import csv
import json
from typing import List, Union

from uni_kie.pdf_to_text.pdf_to_text import KleisterCharityWrapper


class Parser:
    """
    Parses outputs of a model.
    """

    def __init__(self):
        pass

    def parse_single_model_output(self, *args, **kwargs):
        raise NotImplementedError


class KleisterCharityParser(Parser, KleisterCharityWrapper):
    def __init__(self):
        super().__init__()
        KleisterCharityWrapper.__init__(self)

    def parse_single_model_output(self, model_output: str) -> str:
        """
        Assumes that the value for a key is whatever comes after it and before the next key (independent of line breaks)

        The value for some keys will be "null". These are *not* transferred to the expected_output.
        Note that the model output *never includes* the first key.
        """
        model_output = (
            self.gold_key_to_prompt_key[self.gold_keys[0]] + ":" + model_output
        )
        out = []
        for i in range(len(self.gold_keys) - 1):
            key = self.gold_keys[i]
            prompt_key = self.gold_key_to_prompt_key[key]
            next_prompt_key = self.gold_key_to_prompt_key[self.gold_keys[i + 1]]
            value = (
                model_output.split(prompt_key + ":")[1]
                .split(next_prompt_key)[0]
                .strip()
                .replace(" ", "_")
                .replace(":", "_")
            )
            if value != "null":
                out.append(key + "=" + value)

        # last key
        value = (
            model_output.split(self.gold_key_to_prompt_key[self.gold_keys[-1]] + ":")[1]
            .strip()
            .replace(" ", "_")
            .replace(":", "_")
        )

        if value != "null":
            out.append(self.gold_keys[-1] + "=" + value)

        return " ".join(out)

    @staticmethod
    def parse_model_outputs_to_tsv(model_outputs: List[str], model_name: str) -> None:
        """
        Saves into {model_name}_predicted.tsv.

        :param model_outputs: The list of single line strings that come from parse_single_model_output_for_evaluation
        :param model_name: The name of the model
        :return:
        """
        with open(f"{model_name}_predicted.tsv", "w") as tsv_file:
            writer = csv.writer(tsv_file, delimiter="\t", quoting=csv.QUOTE_NONE)
            for model_output in model_outputs:
                writer.writerow([model_output])


class JSONParser(Parser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_single_model_output(model_output: str, gold_keys: List[str]) -> dict:
        """
        Assumes that the value for a key is whatever comes after it and before the next key (independent of line breaks)

        The value for some keys will be "null". These are *not* transferred to the expected_output.
        Note that the model output *never includes* the first key.

        Assumes that model_output is ordered according to gold_keys.
        """
        model_output = gold_keys[0] + ":" + model_output
        out = {}
        for i in range(len(gold_keys) - 1):
            key = gold_keys[i]
            next_key = gold_keys[i + 1]
            value = model_output.split(key + ":")[1].split(next_key)[0].strip()
            if value != "null":
                out[key] = value

        # last key (no next key) but still check if it's null
        value = model_output.split(gold_keys[-1] + ":")[1].strip()
        if value != "null":
            out[gold_keys[-1]] = value

        return out

    @staticmethod
    def parse_model_outputs_to_json(model_outputs: List[dict], model_name: str) -> None:
        """
        Saves into {model_name}_predicted.json.

        :param model_outputs: The list of dicts that come from parse_single_model_output_for_evaluation
        :param model_name: The name of the model
        :return:
        """
        with open(f"{model_name}_predicted.json", "w") as json_file:
            json.dump(model_outputs, json_file)
