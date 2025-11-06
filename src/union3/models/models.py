from pathlib import Path
from union3 import Config, Data, logger


class Model:
    def initialise_from_data(self, data: Data):
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: Config) -> "Model":
        model_path = config.model_path
        if model_path.suffix == ".stan":
            return StanModel(model_path)
        else:
            raise ValueError(f"Unsupported model file extension: {model_path.suffix}")


class StanModel(Model):
    def __init__(self, model_path: Path):
        assert model_path.exists(), f"Model file {model_path} does not exist."
        self.model_path = model_path
        self.model_text = self.model_path.read_text()

        # This is the dictionary of data to be passed to Stan
        # Its different to the global Data obejct because this is transformed into numerical values
        # in arrays, as opposed to dataframes or other structures
        self.data = {}
        logger.info(f"Loaded Stan model from {model_path}.")

    def initialise_from_data(self, data: Data):
        self.data = {}
