from uni_kie.models.model import LargeLanguageModel


class T0pp(LargeLanguageModel):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"T0pp()"

    def predict(self, input: str) -> str:
        return input
