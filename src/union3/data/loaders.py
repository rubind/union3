from pathlib import Path
from typing import Self, TypedDict
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, computed_field
import numpy as np
from union3.config import Config
from union3 import logger
from scipy.special import erf
from scipy.interpolate import interp1d
from astropy.io import fits
import yaml


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
        # selection effects and k-corrections. First, we load the mag_cut file
        mag_cut_file = pl.read_csv(config.data_dir / config.mag_cut_file, separator=",", comment_prefix="#")

        # The mag_cut file has columns sample, kc_file, est_cut_value, and est_cut_sigma
        filtered = filtered.join(mag_cut_file, on="survey", how="left")

        # With the two cuts available for interpolation, we add them in as mobs_cut0 and mobs_cut1
        filtered = add_mobs_cuts(filtered, config.data_dir)

        # TODO: add in the bulk flow load_bulk_flow_data function after figuring out wtf it does
        filtered = add_pecv_and_bulk_flows(filtered, config)

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


def add_mobs_cuts(snia: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    # Each row of this dataframe will have a the survey name, the k-correction file, and two numbers
    # characterising the selection effects: est_mobs_cuts and est_mobs_sigmas.
    k_corrections = {
        kc_file: pl.read_csv(data_dir / f"selection/{kc_file}.csv", separator=",", comment_prefix="#")
        for kc_file in snia["kc_file"].unique().to_list()
    }
    k_correction_cut0_interp = {
        kc_file: interp1d(df["redshift"].to_list(), df["c2"].to_list(), kind="linear")
        for kc_file, df in k_corrections.items()
    }
    k_correction_cut1_interp = {
        kc_file: interp1d(df["redshift"].to_list(), df["c3"].to_list(), kind="linear")
        for kc_file, df in k_corrections.items()
    }

    def compute_mobs_cuts(row: dict) -> dict[str, float]:
        assert row["kc_file"] in k_correction_cut0_interp, f"Unknown kc_file: {row['kc_file']}"
        assert row["kc_file"] in k_correction_cut1_interp, f"Unknown kc_file: {row['kc_file']}"
        assert isinstance(row["z_heliocentric"], (float, int)), "z_heliocentric must be numeric"
        return {
            "mobs_cut0": k_correction_cut0_interp[row["kc_file"]](row["z_heliocentric"]),
            "mobs_cut1": k_correction_cut1_interp[row["kc_file"]](row["z_heliocentric"]),
        }

    return snia.with_columns(
        mb_cuts=pl.struct(["kc_file", "z_heliocentric"]).map_elements(
            compute_mobs_cuts, return_dtype=pl.Struct({"mobs_cut0": pl.Float64, "mobs_cut1": pl.Float64})
        )
    ).unnest("mb_cuts")


def add_pecv_and_bulk_flows(snia: pl.DataFrame, config: Config) -> pl.DataFrame:
    bulk_flows = load_bulk_flow_data(config.data_dir)
    bulk_df, eigenvectors = bulk_flows["redshifts"], bulk_flows["eigenvectors"]
    # From read_and_sample_H0.py line 386, and knowing there are no calibrators, we need to extra pecv
    pec_vel_diag = (
        config.peculiar_velocity_dispersion
        * (5 / np.log(10))
        * (snia["z_cmb"] + 1)
        / (snia["z_cmb"] * (snia["z_cmb"] * 0.5 + 1)) ** 2
    )
    calibrator_names = []
    if config.include_peculiar_velocity_covariance:
        for index, row in snia.iter_rows(named=True):
            # For each supernova, we want to find the closest point in our bulk flow redshift list
            # so that we can find the right set of (top 100) eigenvectors to use for the convariance
            ra: float = row["RA"]  # type: ignore
            dec: float = row["DEC"]  # type: ignore
            z: float = row["z_cmb"]  # type: ignore

            if z < 0.1:
                continue

            distances = (bulk_df["ra"] - ra) ** 2 + (bulk_df["dec"] - dec) ** 2 + 1e6(bulk_df["redshift"] - z) ** 2  # type: ignore
            closest_index = distances.argmin()
            # TODO: figure out wtf is going on with BULK_%03i at line 401

            for bulk_i in range(len(eigenvectors)):
                key = f"BULK_{bulk_i:03d}"


class BulkFlowData(TypedDict):
    redshifts: pl.DataFrame
    eigenvectors: np.ndarray


def load_bulk_flow_data(data_dir: Path) -> BulkFlowData:
    redshifts = pl.read_csv(data_dir / "bulk_flows" / "bulk_flows_redshift.csv")

    eigenvector_file = data_dir / "bulk_flows" / "dominant_evecs.fits"
    with fits.open(eigenvector_file) as hdul:
        eigenvectors = hdul[0].data  # type: ignore

    assert (
        len(eigenvectors) == redshifts.height
    ), f"Mismatch between eigenvectors ({len(eigenvectors)}) and redshifts ({redshifts.height})"

    return {
        "redshifts": redshifts,
        "eigenvectors": eigenvectors,
    }


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
            "ra": "RA",
            "dec": "DEC",
        }
    )


def _filter_snia(df: pl.DataFrame, config: Config):
    weird_sn_file = config.data_dir / "misc/weird_sn.yml"
    assert weird_sn_file.exists(), f"Weird SN file not found at: {weird_sn_file}"
    weird_sn = [str(x) for x in yaml.safe_load(weird_sn_file.read_text())["weird_sn_names"]]

    df_filtered = df.filter(
        pl.col("redshift").is_between(config.filters.min_redshift, config.filters.max_redshift)
        & pl.col("mb").is_between(0, 50)
        & pl.col("color").is_between(config.filters.min_color, config.filters.max_color)
        & pl.col("color_err").is_between(0, config.filters.max_color_uncertainty)
        & pl.col("MWEBV").is_between(0, config.filters.max_MWEBV)
        & pl.col("first_phase").le(config.filters.max_first_phase)
        & pl.col("last_phase").ge(config.filters.min_last_phase)
        & ~pl.col("name").is_in(weird_sn)
        & pl.col("x1_err").is_not_null()
        & ((pl.col("x1").abs() + pl.col("x1_err")) < 5)
    )
    logger.info(f"Filtered to {df_filtered.height} SNe Ia")
    return df_filtered


def impute_snia(df: pl.DataFrame) -> pl.DataFrame:
    """Impute missing values in the SNe Ia dataframe.

    Currently, this function fills missing mass and mass_err values
    with default values.
    """

    return (
        # If z_cmb is missing, we fill it with z_heliocentric if the redshift is > 0.1
        df.with_columns(
            z_cmb=pl.when(pl.col("z_cmb").is_null() & pl.col("z_heliocentric").gt(0.1))
            .then(pl.col("z_heliocentric"))
            .otherwise(pl.col("z_cmb"))
        )
        # Ditto if z_heliocentric is missing, we fill it with z_cmb if the redshift is > 0.1
        .with_columns(
            z_heliocentric=pl.when(pl.col("z_heliocentric").is_null() & pl.col("z_cmb").gt(0.1))
            .then(pl.col("z_cmb"))
            .otherwise(pl.col("z_heliocentric"))
        )
        # Right now, the mass column may be missing from the source files, where
        # generally it would be sourced from the 'lightfile'. If it's not missing
        # it could also have erronenous values. If so, we impute based upon a redshift
        # cut, which is right now hardcoded to z=0.1.
        .with_columns(mass_err=(pl.col("mass_err_lower").abs() * pl.col("mass_err_upper").abs()).sqrt())
        .with_columns(bad_mass=pl.col("mass").is_null() | pl.col("mass").lt(1))
        .with_columns(
            mass=pl.when(pl.col("bad_mass"))
            .then(pl.when(pl.col("z_cmb") < 0.1).then(10.0).otherwise(11.0))
            .otherwise(pl.col("mass")),
            mass_err=pl.when(pl.col("bad_mass")).then(0.1).otherwise(pl.col("mass_err")),
        )
        # The original code has a whole is_calibrator flag, but there are no paramfiles
        # that utilise the distance_ladder key, so this functionality is unused right now.
        # As such, the original code sets has_distmod and distmod to 0.
        # TODO: check to see if I can remove this with David?
        .with_columns(has_distmod=pl.lit(0), distmod=pl.lit(0))
    )
