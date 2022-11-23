import os
import sys

import openai
import requests

from uni_kie.models.model import LargeLanguageModel
from uni_kie.prompts.prompts import STOP_KEY


class GPT3_Davinci(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        # openai.api_key = os.getenv("OPENAI_TOKEN")

        # we are subtracting 256 to leave enough space for the generated text (should never be more than 256 tokens)
        # NOTE: if using TRUNCATE_END, TRUNCATE_START or TRUNCATE_MIDDLE this has to be the smallest of all models to be comparable
        # babbage + curie: 2048 - 256 = 1792
        self.max_input_tokens = 3840
        self.max_generated_tokens = 256

    def __repr__(self):
        return super().__repr__()

    def predict(self, prompt: str) -> str:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            # temperature=0.1,
            # top_p=1,
            max_tokens=self.max_generated_tokens,
            # presence_penalty=-0.75,
            # frequency_penalty=-0.75,
            stop=[STOP_KEY],
        )["choices"][0]["text"]
        return response


class GPT_NeoX(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.api_url = "https://api.textsynth.com"
        self.api_engine = "gptneox_20B"
        self.api_key = os.environ["TEXTSYNTH_API_SECRET_KEY"]
        self.max_input_tokens = 1792
        self.max_generated_tokens = 256

    def __repr__(self):
        return super().__repr__()

    def predict(self, prompt: str) -> str:
        response = requests.post(
            self.api_url + "/v1/engines/" + self.api_engine + "/completions",
            headers={"Authorization": "Bearer " + self.api_key},
            json={
                "prompt": prompt,
                # "temperature": 0,
                # "top_p": 1,
                "max_tokens": self.max_generated_tokens,
                # "presence_penalty": -0.5,
                # "frequency_penalty": -0.5,
                "stop": [STOP_KEY],
            },
        )
        resp = response.json()
        return resp["text"]
