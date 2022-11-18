class AbstractModel:
    def __init__(self):
        pass

    def predict(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class LargeLanguageModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.max_input_tokens: int

    def __repr__(self):
        return super().__repr__

    def predict(self, input: str) -> str:
        raise NotImplementedError
