from loguru import logger
from rich.logging import RichHandler
from union3.config import Config, CosmologyModel
from union3.data import Data
from union3.models.models import Model

logger.configure(handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}])

__all__ = ["logger", "Config", "CosmologyModel", "Data", "Model"]
