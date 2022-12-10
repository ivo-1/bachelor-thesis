import json
import re
from typing import List, Optional, Union

import dateparser
import regex

from uni_kie import create_logger
from uni_kie.kleister_charity_constants import KLEISTER_CHARITY_CONSTANTS

logger = create_logger(__name__)


class Parser:
    """
    Parses outputs of a model.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @staticmethod
    def _parse_money(money: str) -> Optional[str]:
        money = "".join([c for c in money if c.isnumeric() or c == "."])
        money = money.lstrip("0")

        if "." not in money and len(money) > 0:
            money += ".00"

        # if it's an empty string or contains only dots or doesn't contain any numbers
        if (
            money.strip() == ""
            or money.strip(".") == ""
            or not any(c.isnumeric() for c in money)
        ):
            return None

        return money

    @staticmethod
    def _parse_date_to_iso_format(date: str) -> Union[str, None]:
        parsed_date = dateparser.parse(date)
        if parsed_date:
            return parsed_date.strftime("%Y-%m-%d")

    def _dict_to_kleister_charity(
        self, parsed_dict: dict, prompt_keys: List[str]
    ) -> str:
        out = []
        for i in range(len(prompt_keys)):
            gold_key = KLEISTER_CHARITY_CONSTANTS.gold_keys[
                i
            ]  # the gold keys of the KleisterCharity dataset
            prompt_key = prompt_keys[i]  # the prompt key of the prompt

            try:
                value = (
                    parsed_dict[prompt_key]
                    .strip()
                    .replace("\n", " ")
                    .replace(" ", "_")
                    .replace(":", "_")
                )
            except KeyError:  # dict may not contain all keys
                continue

            if gold_key == "report_date":
                value = self._parse_date_to_iso_format(value)

            if value is None or value == "" or value.lower() == "null":
                continue

            out.append(f"{gold_key}={value}")
        return " ".join(out)

    def parse_model_output(
        self, model_output: str, prompt_keys: List[str]
    ) -> Union[str, dict]:
        raise NotImplementedError


class KleisterCharityParser(Parser):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return super().__repr__()

    @staticmethod
    def parse_model_output(model_output: str, prompt_keys: List[str]) -> str:
        """
        Assumes that the value for a key is whatever comes after it and before the next key (independent of line breaks). Also
        assumes that model_output is ordered according to gold_keys.

        If the next key is not in the model output we continue looking for the next key after that one and so on until we
        find a key that is in the model output or we reach the end of the model output.

        The value is then cleaned up by removing line breaks and leading and trailing whitespaces. Also, for the "income"
        and "spending" keys, there is an additional parser which expects some kind of number which, among other things,
        removes leading £ symbol (because that's what the solution expects). If it can't find any number, there won't be
        any value for those keys and they are left out of the final answer.

        Note that any given model output *never includes* the first key hence we add it to the model_output manually.

        The format for Kleister Charity is a single line like so: "{gold_key_1}={value_1} {gold_key_2}={value_2}" with spaces
        separating the key-value pairs. If a value for a key is an empty string or "null", the key-value pair is left out.

        This is important because not every key can be found in the document.
        """
        model_output = prompt_keys[0] + ":" + model_output
        out = []
        gold_keys = None
        if prompt_keys[0] == "Address (post code)":
            gold_keys = KLEISTER_CHARITY_CONSTANTS.SPECIFIC_BASELINE.gold_keys
        else:
            gold_keys = KLEISTER_CHARITY_CONSTANTS.gold_keys
        logger.info("Parsing single model output")
        for i in range(len(prompt_keys)):
            gold_key = gold_keys[i]  # the gold keys of the KleisterCharity dataset
            prompt_key = prompt_keys[i] + ":"
            logger.info(f"Key: {prompt_key}")

            if prompt_key.lower() not in model_output.lower():
                continue

            next_key = None
            for j in range(i + 1, len(prompt_keys)):
                if prompt_keys[j].lower() + ":" in model_output.lower():
                    next_key = prompt_keys[j]
                    break

            # escape parentheses in keys to prevent them from being interpreted as regex groups
            prompt_key_escaped = prompt_key.replace("(", "\(").replace(")", "\)")

            if next_key is not None:
                next_key_escaped = next_key.replace("(", "\(").replace(")", "\)")

                # use regex split to split on prompt_key and find the start of the value
                start = regex.split(
                    prompt_key_escaped, model_output, flags=regex.IGNORECASE
                )[1]

                # use regex split to split on next_key to find the end of the value
                value = regex.split(next_key_escaped, start, flags=regex.IGNORECASE)[0]

            else:
                value = regex.split(
                    prompt_key_escaped, model_output, flags=regex.IGNORECASE
                )[1]
            logger.info(f"Raw value: {value}")
            value = value.strip().replace("\n", " ").replace("  ", " ")

            # remove potential leading and trailing quotation marks
            if (
                value.startswith('"')
                and value.endswith('"')
                or value.startswith("'")
                and value.endswith("'")
            ):
                value = value[1:-1]

            value = (
                value.strip()
            )  # strip again because there may have been spaces just before/after the quotation marks
            logger.info(f"Stripped value: {value}")
            if (
                value.lower() == "null"
                or value == ""
                or value is None
                or value.startswith("null ")
            ):
                continue

            if gold_key == "report_date":
                value = Parser._parse_date_to_iso_format(value)

            if "income" in gold_key or "spending" in gold_key:
                value = Parser._parse_money(value)

            if value is None:
                continue

            value = value.replace(" ", "_").replace(":", "_")
            out.append(f"{gold_key}={value}")
        return " ".join(out)


class DictParser(Parser):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return super().__repr__()

    @staticmethod
    def parse_model_output(model_output: str, prompt_keys: List[str]) -> dict:
        """
        Assumes that the value for a key is whatever comes after it and before the next key (independent of line breaks). Also
        assumes that model_output is ordered according to gold_keys.

        If the next key is not in the model output we continue looking for the next key after that one and so on until we
        find a key that is in the model output or we reach the end of the model output.

        The value is then cleaned up by removing line breaks and leading and trailing whitespaces.

        Note that any given model output *never includes* the first key hence we add it to the model_output manually.

        The format for Kleister Charity is a single line like so: "{gold_key_1}={value_1} {gold_key_2}={value_2}" with spaces
        separating the key-value pairs. If a value for a key is an empty string or "null", the key-value pair is left out.

        This is important because not every key can be found in the document.
        """
        model_output = prompt_keys[0] + ":" + model_output
        out = {}
        for i in range(len(prompt_keys)):
            prompt_key = prompt_keys[i] + ":"
            logger.info(f"Key: {prompt_key}")

            if prompt_key.lower() not in model_output.lower():
                continue

            next_key = None
            for j in range(i + 1, len(prompt_keys)):
                if prompt_keys[j].lower() + ":" in model_output.lower():
                    next_key = prompt_keys[j]
                    break

            prompt_key_escaped = prompt_key.replace("(", "\(").replace(")", "\)")

            if next_key is not None:
                next_key_escaped = next_key.replace("(", "\(").replace(")", "\)")
                start = regex.split(
                    prompt_key_escaped, model_output, flags=regex.IGNORECASE
                )[1]
                value = regex.split(next_key_escaped, start, flags=regex.IGNORECASE)[0]
            else:
                value = regex.split(
                    prompt_key_escaped, model_output, flags=regex.IGNORECASE
                )[1]

            logger.info(f"Raw value: {value}")
            value = value.strip().replace("\n", " ").replace("  ", " ")

            # remove potential leading and trailing quotation marks
            if (
                value.startswith('"')
                and value.endswith('"')
                or value.startswith("'")
                and value.endswith("'")
            ):
                value = value[1:-1]

            value = (
                value.strip()
            )  # strip again because there may have been spaces just before/after the quotation marks

            logger.info(f"Stripped value: {value}")
            if (
                value.lower() == "null"
                or value == ""
                or value is None
                or value.startswith("null ")
            ):
                continue

            parsed_date = Parser._parse_date_to_iso_format(value)
            if parsed_date:
                value = parsed_date

            if "income" in prompt_key.lower() or "spending" in prompt_key.lower():
                value = Parser._parse_money(value)

            if value is None:
                continue

            out[prompt_key[:-1]] = value  # remove trailing colon

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
