import os

import openai

from uni_kie.models.model import LargeLanguageModel


class GPT3_Davinci(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        openai.api_key = os.getenv("OPENAI_TOKEN")

    def __repr__(self):
        return f"GPT_3_Davinci(LargeLanguageModel)"

    def predict(self, prompt: str) -> str:
        response = openai.Completion.create(
            model="text-babbage-001",
            prompt=prompt,
            temperature=0,
            top_p=1,
            max_tokens=60,
            presence_penalty=-0.5,
            frequency_penalty=-0.5,
        )["choices"][0]["text"]
        return response
