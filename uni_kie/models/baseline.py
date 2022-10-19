from uni_kie.models.model import AbstractModel


class BaselineModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"BaselineModel()"

    def predict(self, input: str) -> str:
        return input
