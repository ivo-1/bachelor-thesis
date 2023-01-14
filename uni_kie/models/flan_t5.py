import json
import os

import boto3
import requests
from sagemaker.serializers import JSONSerializer

from uni_kie.models.model import LargeLanguageModel


class FLAN_T5(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.endpoint_url = (
            "https://w5jinsyv9isnuqfg.us-east-1.aws.endpoints.huggingface.cloud"
        )
        self.api_key = os.environ["HUGGINGFACE_API_KEY"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        self.max_input_tokens = (
            1792  # TODO: find the max input tokens (only limited by memory)
        )
        self.max_generated_tokens = 256
        self.temperature = 100  # default: 1
        self.top_p = 0.9  # default: 0.9
        self.top_k = 40  # default: 40
        self.repetition_penalty = 0.0000001  # default: 0.0 (not allowed to be 0.0)

    def __repr__(self):
        return f"Flan_T5(max_input_tokens={self.max_input_tokens}, temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}"

    def predict(self, input: str) -> str:
        data = {
            "inputs": "Once upon a time, there was",
            "parameters": {
                "max_length": int(self.max_generated_tokens),
                "temperature": float(self.temperature),
                # 'top_p': self.top_p,
                # 'top_k': self.top_k,
                # 'repetition_penalty': self.repetition_penalty,
                # 'return_full_text': True,
            },
            "options": {
                "use_cache": False,
            },
        }

        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=data,
        )

        resp = response.json()
        try:
            return resp[0]["generated_text"]
        except KeyError:
            print(resp)
