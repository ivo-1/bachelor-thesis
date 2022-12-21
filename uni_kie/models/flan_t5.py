import json
import os

import boto3
from sagemaker.serializers import JSONSerializer

from uni_kie.models.model import LargeLanguageModel


class FLAN_T5(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.max_input_tokens = (
            768  # TODO: find the max input tokens (only limited by memory)
        )
        self.client = boto3.client("sagemaker-runtime")
        self.endpoint_name = os.getenv("FLAN_T5_ENDPOINT_NAME")
        self.content_type = "application/json"
        self.accept = "application/json"

    def __repr__(self):
        return super().__repr__()

    def predict(self, input: str) -> str:
        payload = {
            "inputs": input,
            "parameters": {
                "max_length": 3,
                "temperature": 0.7,
            },
            "options": {
                "use_cache": False,
            },
        }
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType=self.content_type,
            Accept=self.accept,
            Body=JSONSerializer().serialize(payload),
        )
        return json.loads(response["Body"].read())[0]["generated_text"]
