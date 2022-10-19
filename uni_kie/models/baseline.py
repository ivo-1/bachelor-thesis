from uni_kie.models.model import AbstractModel


class BaselineModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def predict(self, input: str) -> str:
        return input
