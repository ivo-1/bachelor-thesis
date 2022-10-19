class AbstractModel:
    def __init__(self):
        pass

    def predict(self, input: str) -> str:
        raise NotImplementedError


class LargeLanguageModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def predict(self, input: str) -> str:
        return input
