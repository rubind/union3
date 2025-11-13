from pathlib import Path
from union3.data import Data
from union3.config import Config
from loguru import logger
import polars as pl

from union3.plotting import plot_cosmology_constraints

if __name__ == "__main__":
    from rich.logging import RichHandler

    logger.configure(handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}])
    config = Config()
    logger.info(f"Running Unity with base config file: {config.base}")
    logger.info(f"Run settings: {config.model_dump_json(indent=2)}")
    data = Data.from_config(config)

    here = Path(__file__).parent
    output_dir = here.parent.parent / "output"
    samples_file = output_dir / "mcmc_samples.parquet"
    assert samples_file.exists(), f"Samples file not found: {samples_file}"
    samples = pl.read_parquet(samples_file)

    plot_cosmology_constraints(config, samples)
