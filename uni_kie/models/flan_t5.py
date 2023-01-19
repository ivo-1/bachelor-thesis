import json
import os

import boto3
import requests
from sagemaker.serializers import JSONSerializer

from uni_kie import create_logger
from uni_kie.models.model import LargeLanguageModel
from uni_kie.prompts.prompts import STOP_KEY

logger = create_logger(__name__)


class FLAN_T5(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.endpoint_url = (
            "https://rst67e5y0izhf6c5.us-east-1.aws.endpoints.huggingface.cloud"
        )
        self.api_key = os.environ["HUGGINGFACE_API_KEY"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self.max_input_tokens = 1792  # same as neox
        self.max_generated_tokens = 256
        self.temperature = 1  # 0 not possible - instead: use equivalent do_sample=False (-> greedy decoding) - default: 1
        self.top_p = 0.9  # default: 1.0 but 0.9 for comparing to GPT-NeoX
        self.top_k = 40  # default: 50 but 40 for comparing to GPT-NeoX
        self.min_length = 1  # default: 0 but we want at least one token
        self.stop = [STOP_KEY]

    def __repr__(self):
        return f"Flan_T5(max_input_tokens={self.max_input_tokens}, temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k})"

    def predict(self, input: str) -> str:
        data = {
            "inputs": input,
            "parameters": {
                "do_sample": True,  # equivalent to temperature=0
                "min_length": self.min_length,
                "max_length": int(self.max_generated_tokens),
                "temperature": float(self.temperature),
                "top_p": self.top_p,
                "top_k": self.top_k,
            },
        }

        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=data,
        )

        if response.status_code != 200:
            logger.info(f"Error: {response.status_code}")
            logger.info(f"Error: {response.text}")
            raise Exception("Error: " + str(response.status_code))

        resp = response.json()
        try:
            return resp[0]["generated_text"]
        except KeyError:
            print(resp)
            logger.info(f"Error: {resp}")
