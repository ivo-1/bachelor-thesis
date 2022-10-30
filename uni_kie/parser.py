import csv

from uni_kie.constants import KLEISTER_CHARITY


class Parser:
    """
    will parse the output of the model into a tsv used for the evaluation
    """

    def __init__(self):
        pass


class KleisterCharityParser(Parser):
    def __init__(self):
        super().__init__()
        self.gold_keys = KLEISTER_CHARITY.GOLD_KEYS
        self.prompt_key_to_gold_key = KLEISTER_CHARITY.PROMPT_KEY_TO_GOLD_KEY
        self.gold_key_to_prompt_key = KLEISTER_CHARITY.GOLD_KEY_TO_PROMPT_KEY
        # with open('out.tsv', 'r') as tsvfile:
        #     self.reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

    def parse_single_model_output(self, model_output: str) -> str:
        """
        model_output = " WESTCLIFF-ON-SEA\nAddress (post code): SS0 8HX\nAddress (street):47 SECOND AVENUE\nCharity name: " \
                   "Havens Christian Hospice\nCharity number: 1022119\nAnnual income: null\nReport Date (" \
                   "YYYY-MM-DD, ISO8601): 2016-03-31\nAnnual spending: 9415000.00"

        should yield expected_output = "address__post_town=WESTCLIFF-ON-SEA address__postcode=SS0_8HX " \
                              "address__street_line=47_SECOND_AVENUE charity_name=Havens_Christian_Hospice " \
                              "charity_number=1022119 " \
                              "report_date=2016-03-31 spending_annually_in_british_pounds=9415000.00"

        Note that the model output *never includes* the first key.

        Assumes that the value for a key is whatever comes after it and before the next key (independent of line breaks).
        It is possible that the value for some keys is "null". These are *not* transferred to the expected_output.
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
