from typing import Self
import polars as pl
from pydantic import BaseModel, ConfigDict

from union3.config import Config
from union3 import logger


class Data(BaseModel):
    all_supernova: pl.DataFrame
    filtered_supernova: pl.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        all_supernova = _load_snia_lightcurve_fits(config)
        filtered_supernova = _filter_snia(all_supernova, config)
        return cls(
            all_supernova=all_supernova,
            filtered_supernova=filtered_supernova,
        )


def _load_snia_lightcurve_fits(config: Config):
    dfs = [pl.read_parquet(f) for f in (config.data_dir / "supernova_lc_fits").glob("*.parquet")]
    if not dfs:
        raise FileNotFoundError(f"No parquet files found in {config.data_dir}")
    df = pl.concat(dfs, how="diagonal_relaxed")
    logger.info(f"Loaded {df.height} SNe Ia from {config.data_dir}")
    return df.rename(
        {
            "restframemag_0_b": "mb",
            "restframemag_0_b_err": "mb_err",
            "mwebv": "MWEBV",
            "firstphase": "first_phase",
            "lastphase": "last_phase",
        }
    )


def _filter_snia(df: pl.DataFrame, config: Config):
    df_filtered = df.filter(
        pl.col("redshift").is_between(config.filters.min_redshift, config.filters.max_redshift)
        & pl.col("mb").is_between(0, 50)
        & pl.col("color").is_between(config.filters.min_color, config.filters.max_color)
        & pl.col("color_err").is_between(0, config.filters.max_color_uncertainty)
        & pl.col("MWEBV").is_between(0, config.filters.max_MWEBV)
        & pl.col("first_phase").le(config.filters.max_first_phase)
        & pl.col("last_phase").ge(config.filters.min_last_phase)
    )
    logger.info(f"Filtered to {df_filtered.height} SNe Ia")
    return df_filtered
