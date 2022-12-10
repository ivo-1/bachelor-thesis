from uni_kie.models.model import LargeLanguageModel


# TODO: deploy model to ec2 instance and call it here instead of locally
class FLAN_T5(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.max_input_tokens = 768  # 1024 - 256 = 768

    def __repr__(self):
        return super().__repr__()

    def predict(self, input: str) -> str:
        pass
