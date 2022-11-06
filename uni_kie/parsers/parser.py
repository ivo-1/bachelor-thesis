import csv
import json
from typing import List, Union

import dateparser

from uni_kie.kleister_charity_constants import KLEISTER_CHARITY_CONSTANTS


class Parser:
    """
    Parses outputs of a model.
    """

    def __init__(self):
        pass

    def _parse_date_to_iso_format(self, date: str) -> Union[str, None]:
        parsed_date = dateparser.parse(date)
        if parsed_date:
            return parsed_date.strftime("%Y-%m-%d")

    def parse_single_model_output(self, model_output: str, prompt_keys: List[str]):
        raise NotImplementedError


class KleisterCharityParser(Parser):
    def __init__(self):
        super().__init__()

    def parse_single_model_output(
        self, model_output: str, prompt_keys: List[str]
    ) -> str:
        """
        Assumes that the value for a key is whatever comes after it and before the next key (independent of line breaks).
        If the next key is not in the model output we continue looking for the next key after that one and so on until we
        find a key that is in the model output or we reach the end of the model output.

        The value is then cleaned up by removing line breaks and leading and trailing whitespaces.

        The value for some keys will be "null". These are *not* transferred to the expected_output.
        Note that the model output *never includes* the first key hence we add it to the model_output manually.
        """
        model_output = prompt_keys[0] + ":" + model_output
        print(f"model_output:\n{model_output}")
        out = []
        for i in range(len(prompt_keys)):
            gold_key = KLEISTER_CHARITY_CONSTANTS.gold_keys[
                i
            ]  # the gold keys of the KleisterCharity dataset
            prompt_key = prompt_keys[i]

            if prompt_key not in model_output:
                continue

            next_key = None
            for j in range(i + 1, len(prompt_keys)):
                if prompt_keys[j] in model_output:
                    next_key = prompt_keys[j]
                    break

            if next_key is not None:
                value = model_output.split(prompt_key + ":")[1].split(next_key + ":")[0]
            else:
                value = model_output.split(prompt_key + ":")[1]

            value = value.strip().replace(" ", "_").replace(":", "_")

            if gold_key == "report_date":
                value = self._parse_date_to_iso_format(value)

            if value == "null" or value == "" or value is None:
                continue

            out.append(f"{gold_key}={value}")

        print(f"out: {out}")
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
    def parse_single_model_output(model_output: str, prompt_keys: List[str]) -> dict:
        """
        Assumes that the value for a key is whatever comes after it and before the next key (independent of line breaks)

        The value for some keys will be "null". These are *not* transferred to the expected_output.
        Note that the model output *never includes* the first key.

        Assumes that model_output is ordered according to gold_keys.
        """
        model_output = prompt_keys[0] + ":" + model_output
        out = {}
        for i in range(len(prompt_keys)):
            prompt_key = prompt_keys[i]

            if prompt_key not in model_output:
                continue

            next_key = None
            for j in range(i + 1, len(prompt_keys)):
                if prompt_keys[j] in model_output:
                    next_key = prompt_keys[j]
                    break

            if next_key is not None:
                value = model_output.split(prompt_key + ":")[1].split(next_key + ":")[0]
            else:
                value = model_output.split(prompt_key + ":")[1]

            value = value.strip()

            if value == "null" or value == "" or value is None:
                continue

            out[prompt_key] = value

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