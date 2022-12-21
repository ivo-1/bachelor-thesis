import os

import boto3
from sagemaker.huggingface import HuggingFaceModel

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]


# Hub Model configuration. https://huggingface.co/models
hub = {"HF_MODEL_ID": "google/flan-t5-xxl", "HF_TASK": "text2text-generation"}  # xxl

iam_client = boto3.client("iam")

# IAM role
role = iam_client.get_role(RoleName="sagemaker-ivo")["Role"][
    "Arn"
]  # dev/staging: ivoSM, prod: sagemaker-ivo


# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version="4.17.0",  # 4.17
    pytorch_version="1.10.2",  # 1.10.2
    py_version="py38",  # py38
    env=hub,
    role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1,  # number of instances
    instance_type="ml.inf2.6xlarge",  # ec2 instance type
)
