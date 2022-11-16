class AbstractModel:
    def __init__(self):
        pass

    def predict(self, *args, **kwargs) -> str:
        raise NotImplementedError


class LargeLanguageModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.max_input_tokens: int

    def __repr__(self):
        return self.__class__.__name__

    def predict(self, input: str) -> str:
        raise NotImplementedError
