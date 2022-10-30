import csv
import json
from typing import List


class Parser:
    """
    Parses outputs of a model.
    """

    def __init__(self):
        pass

    def parse_single_model_output(self, model_output: str):
        raise NotImplementedError


class KleisterCharityParser(Parser):
    def __init__(self):
        super().__init__()
        self.gold_keys = [
            "address__post_town",
            "address__postcode",
            "address__street_line",
            "charity_name",
            "charity_number",
            "income_annually_in_british_pounds",
            "report_date",
            "spending_annually_in_british_pounds",
        ]
        self.prompt_key_to_gold_key = {
            "Address (post town)": "address__post_town",
            "Address (post code)": "address__post_code",
            "Address (street)": "address__street_line",
            "Charity Name": "charity_name",
            "Charity Number": "charity_number",
            "Annual Income": "income_annually_in_british_pounds",
            "Report Date (YYYY-MM-DD, ISO8601)": "report_date",
            "Annual Spending": "spending_annually_in_british_pounds",
        }
        self.gold_key_to_prompt_key = {
            "address__post_town": "Address (post town)",
            "address__postcode": "Address (post code)",
            "address__street_line": "Address (street)",
            "charity_name": "Charity Name",
            "charity_number": "Charity Number",
            "income_annually_in_british_pounds": "Annual Income",
            "report_date": "Report Date (YYYY-MM-DD, ISO8601)",
            "spending_annually_in_british_pounds": "Annual Spending",
        }

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
        out.append(
            self.gold_keys[-1]
            + "="
            + model_output.split(self.gold_key_to_prompt_key[self.gold_keys[-1]] + ":")[
                1
            ]
            .strip()
            .replace(" ", "_")
            .replace(":", "_")
        )
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
    def __init__(self, gold_keys: List[str]):
        super().__init__()
        self.gold_keys = gold_keys

    def parse_single_model_output(self, model_output: str) -> dict:
        """
        Assumes that the value for a key is whatever comes after it and before the next key (independent of line breaks)

        The value for some keys will be "null". These are *not* transferred to the expected_output.
        Note that the model output *never includes* the first key.
        """
        model_output = self.gold_keys[0] + ":" + model_output
        out = {}
        for i in range(len(self.gold_keys) - 1):
            key = self.gold_keys[i]
            next_key = self.gold_keys[i + 1]
            value = model_output.split(key + ":")[1].split(next_key)[0].strip()
            if value != "null":
                out[key] = value

        # last key
        out[self.gold_keys[-1]] = (
            model_output.split(self.gold_keys[-1] + ":")[1]
            .strip()
            .replace(" ", "_")
            .replace(":", "_")
        )
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
