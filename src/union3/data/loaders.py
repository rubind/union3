import polars as pl

from union3.config import Config
from union3 import logger


def load_snia_lightcurve_fits(config: Config):
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
        }
    )


def filter_snia(df: pl.DataFrame, config: Config):
    df_filtered = df.filter(
        pl.col("redshift").is_between(config.filters.min_redshift, config.filters.max_redshift)
        & pl.col("mb").is_between(0, 50)
        & pl.col("color").is_between(config.filters.min_color, config.filters.max_color)
        & pl.col("color_err").is_between(0, config.filters.max_color_uncertainty)
        & pl.col("MWEBV").is_between(0, config.filters.max_MWEBV)
    )
    logger.info(f"Filtered to {df_filtered.height} SNe Ia")
    return df_filtered
