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
        self.temperature = 1  # default: 1
        self.top_p = 0.9  # default: 1
        self.presence_penalty = 0  # default: 0
        self.frequency_penalty = 0  # default: 0
        self.stop = [STOP_KEY]

    def __repr__(self):
        return f"GPT3_Davinci(max_input_tokens={self.max_input_tokens}, temperature={self.temperature}, top_p={self.top_p}, presence_penalty={self.presence_penalty}, frequency_penalty={self.frequency_penalty})"

    def predict(self, prompt: str) -> str:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            # temperature=self.temperature,
            # top_p=self.top_p,
            max_tokens=self.max_generated_tokens,
            # presence_penalty=self.presence_penalty,
            # frequency_penalty=self.frequency_penalty,
            stop=self.stop,
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
        self.temperature = 1  # default: 1
        self.top_p = 0.9  # default: 0.9
        self.top_k = 40  # default: 40
        self.presence_penalty = 0  # default: 0
        self.frequency_penalty = 0  # default: 0
        self.stop = [STOP_KEY]

    def __repr__(self):
        return f"GPT_NeoX(max_input_tokens={self.max_input_tokens}, temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, presence_penalty={self.presence_penalty}, frequency_penalty={self.frequency_penalty})"

    def predict(self, prompt: str) -> str:
        response = requests.post(
            self.api_url + "/v1/engines/" + self.api_engine + "/completions",
            headers={"Authorization": "Bearer " + self.api_key},
            json={
                "prompt": prompt,
                # "temperature": self.temperature,
                # "top_p": self.top_p,
                "max_tokens": self.max_generated_tokens,
                # "presence_penalty": self.presence_penalty,
                # "frequency_penalty": self.frequency_penalty,
                "stop": self.stop,
            },
        )
        resp = response.json()
        return resp["text"]
