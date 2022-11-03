import os
import sys

import openai
import requests

from uni_kie.models.model import LargeLanguageModel


class GPT3_Davinci(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        # openai.api_key = os.getenv("OPENAI_TOKEN")
        # TODO: change this to 4096-256 = 3840 when using davinci
        self.max_input_tokens = 768

    def __repr__(self):
        return f"GPT_3_Davinci(LargeLanguageModel)"

    def predict(self, prompt: str) -> str:
        response = openai.Completion.create(
            model="text-babbage-001",
            prompt=prompt,
            temperature=0,
            top_p=1,
            max_tokens=256,
            presence_penalty=-0.5,
            frequency_penalty=-0.5,
        )["choices"][0]["text"]
        return response


class GPT_NeoX(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.api_url = "https://api.textsynth.com"
        self.api_engine = "gptneox_20B"
        self.api_key = os.environ["TEXTSYNTH_API_SECRET_KEY"]
        self.max_generated_tokens = 256
        self.max_input_tokens = 768

    def __repr__(self):
        return f"GPT_NeoX(LargeLanguageModel)"

    def predict(self, prompt: str) -> str:
        response = requests.post(
            self.api_url + "/v1/engines/" + self.api_engine + "/completions",
            headers={"Authorization": "Bearer " + self.api_key},
            json={
                "prompt": prompt,
                "temperature": 0,
                "top_p": 0.01,
                "max_tokens": self.max_generated_tokens,
                "presence_penalty": -0.5,
                "frequency_penalty": -0.5,
            },
        )
        resp = response.json()
        return resp["text"]
