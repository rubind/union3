from typing import Self, TypedDict
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, computed_field
import numpy as np
from union3.config import Config
from union3 import logger
from scipy.special import erf


class Data(BaseModel):
    all_supernova: pl.DataFrame = Field(exclude=True)
    filtered_supernova: pl.DataFrame = Field(exclude=True)

    # Extra fields for applying simpsons rule on redshift integrations
    redshifts_sort_fill: list[float]
    unsort_inds: list[int]
    nzadd: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def num_supernova(self) -> int:
        return self.filtered_supernova.height

    @classmethod
    def from_config(cls, config: Config) -> Self:
        all_supernova = _load_snia_lightcurve_fits(config)
        filtered = _filter_snia(all_supernova, config).pipe(impute_snia)

        # Each supernova is associated with a survey, and each survey has its own
        # selection effects, characterised in a few ways. First, we need to load the mag_
        mag_cut_file = pl.read_csv(config.data_dir / config.mag_cut_file, separator=",", comment_prefix="#")

        # The mag_cut file has columns sample, kc_file, est_cut_value, and est_cut_sigma
        # We'll join the last two on in a moment, but we also want to load in all the kc_files
        # in one go.
        filtered = filtered.join(
            mag_cut_file.select(["sample", "est_mobs_cuts", "est_mobs_sigmas"]), on="survey", how="left"
        )

        # TODO: port helper_functions.get_kcorrect_ifns (164) and interpolate to the redshift-specific values (287)

        extra_redshifts = _get_redshifts(filtered["z_cmb"].to_list())
        p_high_mass = 0.5 * (  # noqa: F841
            1.0 + erf((np.array(filtered["mass"]) - 10.0) / (np.sqrt(2.0) * np.array(filtered["mass_err"])))
        )
        # TODO: p_high_mass

        return cls(
            all_supernova=all_supernova,
            filtered_supernova=filtered,
            **extra_redshifts,
        )


class RedshiftResults(TypedDict):
    redshifts_sort_fill: list[float]
    unsort_inds: list[int]
    nzadd: int


def _get_redshifts(redshifts: list[float]) -> RedshiftResults:
    """Create a redshift array using both data redshifts
    and appended redshift values to create a smooth array
    that can be used for simpson's rule integration.

    Parameters:
        redshifts: List of redshift values from data, typically taken from the z_cmb column.

    Returns:
        A tuple of numpy arrays, being:
        - The redshift array, with extra midpoint values added.
        - An index dereferencing array, mapping the input+added redshifts arrays
        - The number of appended redshift values in un-midpoint-added redshift array.
    """
    assert len(redshifts) > 0, "Redshift list must not be empty."

    appended_redshifts = np.arange(0.0, max(redshifts), 0.1)  # TODO: Ask David why this was originally 2.5 and not max
    combined_redshifts = np.concatenate((np.array(redshifts), appended_redshifts))
    # TODO: I dont follow why we wouldnt sort this immediately

    sort_inds = list(np.argsort(combined_redshifts))
    unsort_inds = [sort_inds.index(i) for i in range(len(combined_redshifts))]

    tmp_redshifts = np.sort(combined_redshifts)
    redshifts_sort_fill = np.sort(np.concatenate((tmp_redshifts, 0.5 * (tmp_redshifts[1:] + tmp_redshifts[:-1]))))

    return {
        "redshifts_sort_fill": redshifts_sort_fill.tolist(),
        "unsort_inds": unsort_inds,
        "nzadd": len(appended_redshifts),
    }


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


def impute_snia(df: pl.DataFrame) -> pl.DataFrame:
    """Impute missing values in the SNe Ia dataframe.

    Currently, this function fills missing mass and mass_err values
    with default values.
    """
    # Right now, the mass column may be missing from the source files, where
    # generally it would be sourced from the 'lightfile'. If it's not missing
    # it could also have erronenous values. If so, we impute based upon a redshift
    # cut, which is right now hardcoded to z=0.1.

    df.with_columns(mass_err=(pl.col("mass_err_lower").abs() * pl.col("mass_err_upper").abs()).sqrt()).with_columns(
        bad_mass=pl.col("mass").is_null() | pl.col("mass").lt(1)
    ).with_columns(
        mass=pl.when(pl.col("bad_mass"))
        .then(pl.when(pl.col("z_cmb") < 0.1).then(10.0).otherwise(11.0))
        .otherwise(pl.col("mass")),
        mass_err=pl.when(pl.col("bad_mass")).then(0.1).otherwise(pl.col("mass_err")),
    )

    return df
