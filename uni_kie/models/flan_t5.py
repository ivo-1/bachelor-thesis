import boto3
from sagemaker.huggingface import HuggingFaceModel

from uni_kie.models.model import LargeLanguageModel


# TODO: deploy model to ec2 instance and call it
class FLAN_T5(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.max_input_tokens = 768  # 1024 - 256 = 768

        iam_client = boto3.client("iam")

        # get this role 'arn:aws:iam::658875237566:role/SagemakerExecution')
        role = iam_client.get_role(RoleName="SagemakerExecution")["Role"]["Arn"]
        # Hub Model configuration. https://huggingface.co/models
        hub = {"HF_MODEL_ID": "google/flan-t5-xxl", "HF_TASK": "text2text-generation"}

        # create Hugging Face Model Class
        huggingface_model = HuggingFaceModel(
            transformers_version="4.17.0",
            pytorch_version="1.10.2",
            py_version="py38",
            env=hub,
            role=role,
        )

        # deploy model to SageMaker Inference
        self.predictor = huggingface_model.deploy(
            initial_instance_count=1,  # number of instances
            instance_type="ml.m5.xlarge",  # ec2 instance type
        )

    def __repr__(self):
        return super().__repr__()

    def predict(self, input: str) -> str:
        return self.predictor.predict({"inputs": input})
