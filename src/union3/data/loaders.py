from collections import defaultdict
import json
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
import polars.selectors as cs

from union3.data.uncertainties import (
    add_electron_scattering_uncertainties,
    add_intergalactic_extinction_uncertainties,
    add_MBEBV_uncertainties,
    rescale_uncertainties,
)


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
        # We start by loading in all the data we should need
        all_supernova = _load_snia_lightcurve_fits(config)

        # Each supernova is associated with a survey, and each survey has its own
        # selection effects and k-corrections. First, we load the mag_cut file
        # The mag_cut file has columns sample, kc_file, est_cut_value, and est_cut_sigma
        mag_cut_file = pl.read_csv(config.data_dir / config.mag_cut_file, comment_prefix="#")

        # Calibration uncertainties can be scaled up or down, and this file is how we do so
        calibration_uncertaintes = load_calibration_uncertainties(config)

        snia = (
            impute_snia(all_supernova, config)
            .pipe(_filter_snia, config)
            .pipe(flag_weird_supernova)
            .pipe(determine_calibrators, config)
            .pipe(add_lensing_bias, config)
            .pipe(add_photoz_errors)
            .join(mag_cut_file, on="survey", how="left")
            .pipe(add_mobs_cuts, config.data_dir)
            .pipe(add_pecv_and_bulk_flow_uncertainties, config)
            .pipe(add_MBEBV_uncertainties, config)
            .pipe(add_intergalactic_extinction_uncertainties, config)
            .pipe(add_electron_scattering_uncertainties, config)
            # TODO: add in Landolt Smith bins (get_dparam_dzps)
            # Now we rescale the uncerts based on the calibration uncertainty scaling factors
            .pipe(rescale_uncertainties, calibration_uncertaintes)
            .pipe(remap_x1, config)
            # At this point we should be done with all the deriv_ columns, having
            # processed them into uncertainty_ columns, and can drop them.
            .drop(cs.starts_with("deriv_") | cs.starts_with("instrument|"))
        )

        extra_redshifts = _get_redshifts(snia["z_cmb"].to_list())
        p_high_mass = 0.5 * (  # noqa: F841
            1.0 + erf((np.array(snia["mass"]) - 10.0) / (np.sqrt(2.0) * np.array(snia["mass_err"])))
        )
        # TODO: p_high_mass

        # TODO: photoz_inds need to be determined.
        return cls(
            all_supernova=all_supernova,
            filtered_supernova=snia,
            **extra_redshifts,
        )


def remap_x1(snia: pl.DataFrame, config: Config) -> pl.DataFrame:
    intercept, slope = config.remap_x1_intercept, config.remap_x1_slope
    if intercept == 0.0 and slope == 0.0:
        return snia
    logger.info(f"Remapping x1 with intercept {intercept} and slope {slope}.")

    x1 = pl.col("x1")
    return (
        snia.with_columns(
            x1=x1 + intercept * x1**2 + slope * x1**3,
            new_x1_slope=1 + 2 * intercept * x1 + 3 * slope * x1**2,
        )
        .with_columns(
            cov_mb_x1=pl.col("cov_mb_x1") * pl.col("new_x1_slope"),
            cov_x1_color=pl.col("cov_x1_color") * pl.col("new_x1_slope"),
            cov_x1_x1=pl.col("cov_x1_x1") * pl.col("new_x1_slope") ** 2,
        )
        .drop("new_x1_slope")
    )


def flag_weird_supernova(snia: pl.DataFrame) -> pl.DataFrame:
    """Flags any supernova that have weird properties."""

    snia = snia.with_columns(
        h_resid=pl.col("mB")
        + 19.1
        + 0.13 * pl.col("x1")
        - 3.0 * pl.col("color")
        - (5 * (pl.col("z_cmb") * (1 + pl.col("z_heliocentric"))).log10() + 42.9)
    ).with_columns(is_weird=(pl.col("h_resid").abs() > 2) | (pl.col("color") > 1.0) | (pl.col("color") < -0.3))

    for row in snia.filter(pl.col("is_weird")).select("name", "x1", "color").to_dicts():
        logger.warning(f"Flagged weird supernova: {row['name']} with stretch {row['x1']} and color {row['color']}")

    return snia


def load_calibration_uncertainties(config: Config) -> dict[str, float]:
    calib_file = config.data_dir / config.calibration_uncertainties_file
    calib_df = pl.read_csv(calib_file).with_columns(
        key=(pl.col("type") + "_" + pl.col("subtype").fill_null("")).str.strip_chars("_")
    )
    logger.info(f"Loaded calibration uncertainties from {calib_file}, providing {calib_df.height} entries.")

    # Unlike the other systematics which have an extra key on the supernova name,
    # the calibration uncertainty files are generic and grouped by both type and subtype.
    interim = dict(zip(calib_df["key"], calib_df["value"]))

    # Following read_and_sample_H0.py line 135+, we add extra entries to this dictionary
    extra = {
        "MWEBV_multnorm": 1.0,
        "MWEBV_addnorm": 1.0,
        "electron_scattering": 1.0,
        "IG_extinction": 1.0,
        "lensing_bias": 1.0,
    }

    combined = {**interim, **extra}
    logger.info(f"Calibration uncertainties are: {json.dumps(combined, indent=2)}")
    return combined


def add_lensing_bias(snia: pl.DataFrame, config: Config) -> pl.DataFrame:
    lensing_bias_file = config.data_dir / config.lensing_bias_file
    if config.use_lensing_file:
        lensing_df = pl.read_csv(lensing_bias_file)
        logger.info(f"Loaded lensing bias from {lensing_bias_file}, providing {lensing_df.height} entries.")
        redshifts = snia["z_cmb"].to_list()
        lensing_bias = interp1d(lensing_df["redshift"].to_list(), lensing_df["mag"].to_list(), kind="linear")(redshifts)
        snia = snia.with_columns(uncertainty_mB_lensing_bias=pl.Series(lensing_bias))
    else:
        logger.info(f"Adding lensing bias using dispersion 0.5*({config.lensing_dispersion}mag * redshift)**2.")
        snia = snia.with_columns(uncertainty_mB_lensing_bias=0.5 * (config.lensing_dispersion * pl.col("z_cmb") ** 2))

    return snia


def add_photoz_errors(snia: pl.DataFrame) -> pl.DataFrame:
    if "photoz_mean" not in snia.columns:
        logger.info("No photo-z supernova found, skipping photo-z error addition.")
        return snia

    logger.info(
        f"Adding photo-z error terms for {snia.filter(pl.col("photoz_mean").is_not_null()).height} photo-z supernova."
    )
    return snia.with_columns(
        uncertainty_mB_photoz_sys=pl.when(pl.col("photoz_mean").is_not_null())
        .then(pl.col("deriv_Redshift_dmB/dP"))
        .otherwise(None),
        uncertainty_x1_photoz_sys=pl.when(pl.col("photoz_mean").is_not_null())
        .then(pl.col("deriv_Redshift_ds/dP"))
        .otherwise(None),
        uncertainty_color_photoz_sys=pl.when(pl.col("photoz_mean").is_not_null())
        .then(pl.col("deriv_Redshift_dc/dP"))
        .otherwise(None),
    )


def determine_calibrators(snia: pl.DataFrame, config: Config) -> pl.DataFrame:
    if config.distance_ladder_file is None:
        return snia.with_columns(is_calibrator=pl.lit(False), distmod=pl.lit(0), has_distmod=pl.lit(0))

    # The calibrators are determined from the distance ladder file, which has N columns, the first two being
    # the name and the distmod itself. The rest of the columns are the diag variance and then all the offdiag terms
    dist_ladder_file = config.data_dir / config.distance_ladder_file
    distance_ladder = (
        pl.read_csv(dist_ladder_file, has_header=False)
        .rename({"column_1": "name", "column_2": "distmod", "column_3": "distmod_err"})
        .select(pl.all().name.map(lambda name: name.replace("column", "distmod_err_offdiag")))
    )
    logger.info(f"Loaded distance ladder from {dist_ladder_file}, providing {distance_ladder.height} calibrators.")
    snia = snia.join(distance_ladder, on="name", how="left").with_columns(
        is_calibrator=pl.col("distmod").is_not_null(),
        has_distmod=pl.when(pl.col("distmod").is_not_null()).then(1).otherwise(0),
        distmod=pl.col("distmod").fill_null(0),  # TODO: maybe we should do the awkward imputation right before stan?
    )

    distmod_err_cols = [col for col in snia.columns if col.startswith("distmod_err")]
    renamed = {
        f"uncertainty_mB_distmod_{k.removeprefix('distmod_err_')}": pl.when(pl.col("is_calibrator"))
        .then(pl.col(k))
        .otherwise(pl.lit(0))
        for k in distmod_err_cols
    }
    snia = snia.with_columns(**renamed)

    return snia


def add_mobs_cuts(snia: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    """With the two cuts available for interpolation, we add them in as mobs_cut0 and mobs_cut1"""
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


def add_pecv_and_bulk_flow_uncertainties(snia: pl.DataFrame, config: Config) -> pl.DataFrame:
    logger.info("Adding peculiar velocity dispersion and bulk flow data.")
    bulk_flows = load_bulk_flow_data(config.data_dir)
    bulk_df, eigenvectors = bulk_flows["redshifts"], bulk_flows["eigenvectors"]

    # From read_and_sample_H0.py line 386, and knowing there are no calibrators, we need to extra pecv
    snia = snia.with_columns(
        pec_vel_on_diag=pl.when(pl.col("is_calibrator"))
        .then(0.0)
        .otherwise(
            config.peculiar_velocity_dispersion
            * (5 / np.log(10))
            * (pl.col("z_cmb") + 1)
            / (pl.col("z_cmb") * (pl.col("z_cmb") * 0.5 + 1)) ** 2
        )
    )

    # Each supernova has calibration/calibrator uncertainty, and we want to keep track of
    # the name of the SN and a key associated with each systematic name. In our case,
    # this will be the 100 bulk flow eigenvectors.
    d_mBx1c_dcalib: dict[str, dict[str, str | float]] = defaultdict(dict)
    final_pecvs: dict[str, float] = {}
    if config.include_peculiar_velocity_covariance:
        for row in snia.filter(pl.col("z_cmb") < 0.1).iter_rows(named=True):
            # For each supernova, we want to find the closest point in our bulk flow redshift list
            # so that we can find the right set of (top 100) eigenvectors to use for the convariance
            ra: float = row["RA"]  # type: ignore
            dec: float = row["DEC"]  # type: ignore
            z: float = row["z_cmb"]  # type: ignore
            name: str = row["name"]  # type: ignore

            distances = (bulk_df["ra"] - ra) ** 2 + (bulk_df["dec"] - dec) ** 2 + 1e6 * (bulk_df["z"] - z) ** 2
            # This is the closest 100 bulk flow eigenvectors that we will use as calibration systematics
            eigenvector = eigenvectors[distances.arg_min()]

            for bulk_i in range(len(eigenvector)):
                key = f"BULK_{bulk_i:03d}"
                d_mBx1c_dcalib[name][key] = eigenvector[bulk_i]

            # Now that we've done the bulk flow eigenvector systematics, we also want to add in
            # the corr_redshift_sys, which I assume is either corrected or correlated redshift systematic.
            d_mBx1c_dcalib[name]["corr_redshift_sys"] = (3.3e-5) * (5 / np.log(10)) * (1.0 + z) / (z * (1 + 0.5 * z))
            # We also have
            total_bulk_quad = np.sum(eigenvector**2)
            total_pec_v_on_diag = np.clip(row["pec_vel_on_diag"] - total_bulk_quad, 0, 100)
            final_pecvs[name] = total_pec_v_on_diag
            logger.debug(f"SN {name}: {total_bulk_quad=}, {total_pec_v_on_diag=}")

    # Turn d_mBx1c_dcalib into a dataframe to join back on
    if d_mBx1c_dcalib:
        systematics = pl.DataFrame(list(d_mBx1c_dcalib.values())).select(
            pl.all().name.prefix("uncertainty_mB_"), pl.Series(list(d_mBx1c_dcalib.keys())).alias("name")
        )
        snia = snia.join(systematics, on="name", how="left")

    # The pec_vel_on_diag needs to be overwritten with the final_pecvs values if theres a value for that row
    snia = snia.with_columns(
        pec_vel_on_diag=pl.when(pl.col("name").is_in(final_pecvs.keys()))
        .then(pl.col("name").replace_strict(final_pecvs, default=None).cast(pl.Float64))
        .otherwise(pl.col("pec_vel_on_diag"))
    ).with_columns(mbmb_var=pl.col("mbmb_var") + pl.col("pec_vel_on_diag"))

    return snia


class BulkFlowData(TypedDict):
    redshifts: pl.DataFrame
    eigenvectors: np.ndarray


def load_bulk_flow_data(data_dir: Path) -> BulkFlowData:
    redshifts = pl.read_csv(data_dir / "bulk_flows" / "bulk_flows_redshift.csv")

    eigenvector_file = data_dir / "bulk_flows" / "dominant_evecs.fits"
    with fits.open(eigenvector_file) as hdul:
        eigenvectors = hdul[0].data.T  # type: ignore

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
            "restframemag_0_b": "mB",
            "restframemag_0_b_err": "mB_err",
            "covrestframemag_0_bx1": "cov_mb_x1",
            "covcolorrestframemag_0_b": "cov_mb_color",
            "covx1x1": "cov_x1_x1",
            "covcolorx1": "cov_color_x1",
            "covcolorcolor": "cov_color_color",
            "mwebv": "MWEBV",
            "firstphase": "first_phase",
            "lastphase": "last_phase",
            "ra": "RA",
            "dec": "DEC",
            "cluster": "in_cluster",
            "p_spike": "photo_pspike",
            "spikez": "photo_spikez",
            "photoz_unc": "photo_sigz",
            "photoz_mean": "photo_z0",
        },
        strict=False,
    ).sort("name")


def _filter_snia(df: pl.DataFrame, config: Config):
    if config.weird_sn_file is None:
        weird_sn = []
        logger.info("No weird SN file provided, skipping weird SN exclusion.")
    else:
        weird_sn_file = config.data_dir / config.weird_sn_file
        weird_sn = [str(x) for x in yaml.safe_load(weird_sn_file.read_text())["weird_sn_names"]]
        logger.info(f"Loaded {len(weird_sn)} weird SN names from {weird_sn_file}.")

    df_filtered = df.filter(
        pl.col("redshift").is_between(config.filters.min_redshift, config.filters.max_redshift)
        & pl.col("mB").is_between(0, 50)
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


def impute_snia(df: pl.DataFrame, config: Config) -> pl.DataFrame:
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
        .with_columns(
            bad_mass=pl.col("mass").is_null()
            | pl.col("mass").lt(1)
            | pl.col("mass_err").is_null()
            | pl.col("mass_err").le(0)
            | pl.col("mass_err").is_infinite()
        )
        .with_columns(
            mass=pl.when(pl.col("bad_mass"))
            .then(pl.when(pl.col("z_cmb") < 0.1).then(10.0).otherwise(11.0))
            .otherwise(pl.col("mass")),
            mass_err=pl.when(pl.col("bad_mass")).then(0.1).otherwise(pl.col("mass_err")),
        )
        # Add in mbmb_var from mb_err and the lensing dispersion
        .with_columns(mbmb_var=pl.col("mB_err") ** 2 + (pl.col("z_cmb") * config.lensing_dispersion) ** 2)
    )
