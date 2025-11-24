from collections import defaultdict
import json
from pathlib import Path
from typing import Literal, Self, TypedDict
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, computed_field
import numpy as np
from union3 import Config, CosmologyModel, logger
from scipy.special import erf
from scipy.interpolate import interp1d
from astropy.io import fits
import yaml
import polars.selectors as cs
from astropy.cosmology import FlatLambdaCDM

from union3.data.uncertainties import (
    add_electron_scattering_uncertainties,
    add_intergalactic_extinction_uncertainties,
    add_MBEBV_uncertainties,
    add_landolt_smith_uncertainties,
    rescale_uncertainties,
)


class BAOCMB_Omw0wa(TypedDict):
    mean: np.ndarray
    cov: np.ndarray


class RedshiftSimps(TypedDict):
    redshifts_sort_fill: list[float]
    unsort_inds: list[int]
    nzadd: int


class RedshiftBins(TypedDict):
    zbins: np.ndarray  # 1D array of shape (n_bins,)
    dmu_dbin: np.ndarray  # 2D array of shape (n_sne, n_bins)
    dmudz_dbin: np.ndarray  # 2D array of shape (n_sne, n_bins)


class Data(BaseModel):
    all_supernova: pl.DataFrame = Field(exclude=True)
    filtered_supernova: pl.DataFrame = Field(exclude=True)
    samples: pl.DataFrame = Field(exclude=True)

    # Represent systematics in the order of the dataframes above (sorted by name)
    systematic_names: list[str] = Field(exclude=True)
    systematics: list[np.ndarray] = Field(exclude=True)

    # Extra fields for applying simpsons rule on redshift integrations
    redshift_simps: RedshiftSimps

    # For binned mu cosmology models
    redshift_bins: RedshiftBins

    redshift_coeffs: np.ndarray  # Redshift coefficients for each supernova
    bao_cmb_omw0wa: BAOCMB_Omw0wa  # mean and covariance matrix for BAO+CMB Omw0wa constraints

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def num_supernovae(self) -> int:
        return self.filtered_supernova.height

    @computed_field
    @property
    def obs_mBx1c(self) -> list[np.ndarray]:
        """Returns the observed mB, x1, color in a vector, one element per supernova."""
        return [
            np.array(x).astype(np.float64)
            for x in self.filtered_supernova.select(["mB", "x1", "color"]).to_numpy().tolist()
        ]

    @computed_field
    @property
    def obs_mBx1c_cov(self) -> list[np.ndarray]:
        """Returns the observed covariance matrices as a 3x3 matrix, one per supernova."""
        # Note that we have both the original columns (like cov_mBmB), plus the extra uncertainty
        # that comes from cov_mBmB_* columns (etc). We need to sum these together.
        snia = self.filtered_supernova
        mBmB = snia.select(pl.sum_horizontal(cs.starts_with("cov_mBmB"))).to_numpy().flatten()
        mBx1 = snia.select(pl.sum_horizontal(cs.starts_with("cov_mBx1"))).to_numpy().flatten()
        mBc = snia.select(pl.sum_horizontal(cs.starts_with("cov_mBc"))).to_numpy().flatten()
        x1x1 = snia.select(pl.sum_horizontal(cs.starts_with("cov_x1x1"))).to_numpy().flatten()
        x1c = snia.select(pl.sum_horizontal(cs.starts_with("cov_x1c"))).to_numpy().flatten()
        cc = snia.select(pl.sum_horizontal(cs.starts_with("cov_cc"))).to_numpy().flatten()
        cov_mBx1c = []
        for i in range(len(mBmB)):
            cov_matrix = np.array(
                [
                    [mBmB[i], mBx1[i], mBc[i]],
                    [mBx1[i], x1x1[i], x1c[i]],
                    [mBc[i], x1c[i], cc[i]],
                ]
            )
            cov_mBx1c.append(cov_matrix.astype(np.float64))
        return cov_mBx1c

    @computed_field
    @property
    def photoz_uncertainty_dz(self) -> list[np.ndarray]:
        cols = ["dz_uncertainty_photoz_mB", "dz_uncertainty_photoz_x1", "dz_uncertainty_photoz_c"]
        if cols[0] not in self.filtered_supernova.columns:
            return []
        return [np.array(x).astype(np.float64) for x in self.filtered_supernova.select(cols).to_numpy().tolist()]

    @classmethod
    def from_config(cls, config: Config) -> Self:
        # We start by loading in all the data we should need
        all_supernova = _load_snia_lightcurve_fits(config)

        # Each supernova is associated with a survey, and each survey has its own
        # selection effects and k-corrections. First, we load the mag_cut file
        # The mag_cut file has columns sample, kc_file, est_cut_value, and est_cut_sigma
        mag_cut_file = pl.read_csv(config.data_dir / config.mag_cut_file, comment_prefix="#")

        # Calibration uncertainties can be scaled up or down, and this file is how we do so
        calib_uncert, calib_standards = load_calibration_uncertainties(config)

        # Apply a billion functions to filter, augment, and add uncertainties to the supernova data
        snia = get_filtered_and_augmented_snia(all_supernova, config, mag_cut_file, calib_uncert, calib_standards)

        # At this point we have a massive dataframe with all the supernova we want to use, their uncertainties,
        # and a ton of computed systematics hidden under the `uncertainty_` prefix. We now need to prepare
        # the data into something a bit more numerically friendly for Stan or whatever our fitting tool is.

        # The stan model wants array[n_sne] matrix[3, n_calib] d_mBx1c_d_calib ;
        # So each SN wants a 3 x n_calib matrix of derivatives. Those are found below
        systematic_names, systematics = condense_systematics(snia)

        samples = (
            snia.select("survey", "sample_index", "est_mobs_cuts", "est_mobs_sigmas").unique().sort("sample_index")
        )
        redshift_simps = _get_redshifts(snia["z_cmb"].to_list())
        redshift_coeffs = _get_redshift_coeffs(snia, config)
        bao_cmb_omw0wa = _get_bao_cmb_omw0wa(config)
        snia, redshift_bins = _get_redshift_bins(snia, config)

        result = cls(
            all_supernova=all_supernova,
            filtered_supernova=snia,
            samples=samples,
            systematics=systematics,
            systematic_names=systematic_names,
            redshift_coeffs=redshift_coeffs,
            bao_cmb_omw0wa=bao_cmb_omw0wa,
            redshift_bins=redshift_bins,
            redshift_simps=redshift_simps,
        )

        # TODO: REMOVE TEMP CODE
        from union3.data.validation import get_the_data, expected_names

        missing_supernova = sorted(list(set(expected_names) - set(snia["name"].to_list())))
        if missing_supernova:
            logger.warning(f"The following expected supernova are missing after filtering: {missing_supernova}")
        snia_which_should_be_dropped = sorted(list(set(snia["name"].to_list()) - set(expected_names)))
        if snia_which_should_be_dropped:
            logger.warning(
                f"The following supernova are present but not expected and will be dropped: {snia_which_should_be_dropped}"
            )
        # We should use this expected_names to sort the dataframe so we can do direct comparisons against the data vectors
        snia = (
            snia.with_columns(
                pl.col("name")
                .map_elements(lambda n: expected_names.index(n), return_dtype=pl.Int64)
                .alias("expected_index")
            )
            .sort("expected_index")
            .drop("expected_index")
        )
        result.filtered_supernova = snia

        # Invoke the model fields to ensure they generate without error
        result.obs_mBx1c
        result.obs_mBx1c_cov

        the_data = get_the_data()
        assert np.allclose(snia["mB"], the_data["mB_list"])
        assert np.allclose(snia["x1"], the_data["x1_list"])
        assert np.allclose(snia["color"], the_data["c_list"])
        assert np.allclose(snia["z_cmb"], the_data["z_CMB_list"])
        assert np.allclose(snia["z_heliocentric"], the_data["z_helio_list"])
        assert np.allclose(snia["has_distmod"], the_data["has_distmod"])
        covs = result.obs_mBx1c_cov
        for sev in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
            logger.info(f"Checking covariances at sev={sev}")
            for davids_index in range(len(covs)):
                assert np.allclose(covs[davids_index], the_data["mBx1c_cov_list"][davids_index], atol=sev)
        for davids_index in range(snia.height):
            if abs(snia["mass_err"][davids_index] - the_data["mass_err"][davids_index]) > 1e-6:
                logger.error(
                    f"mass_err mismatch for SN {snia['name'][davids_index]}: {snia['mass_err'][davids_index]} vs {the_data['mass_err'][davids_index]}"
                )
        assert np.allclose(snia["mass"], the_data["mass"])
        assert np.allclose(snia["mass_err"], the_data["mass_err"])
        assert np.allclose(snia["in_cluster"], the_data["in_cluster"])
        assert np.allclose(snia["mobs_cut0"], the_data["mobs_cut0"])
        assert np.allclose(snia["mobs_cut1"], the_data["mobs_cut1"])

        # Reorder my samples to match Davids ordering
        sample_names = [x.split("//")[-1].split("_v1")[0].lower() for x in the_data["sample_names"]]
        samples2 = samples.sort(
            pl.col("survey").map_elements(lambda n: sample_names.index(n.lower()), return_dtype=pl.Int64)
        )
        assert np.allclose(samples2["est_mobs_cuts"], the_data["est_mobs_cuts"])
        assert np.allclose(samples2["est_mobs_sigmas"], the_data["est_mobs_sigmas"])

        systematic_names, systematics = condense_systematics(snia)

        # Check the number of systematics agree
        davids_sys = []
        for name in the_data["calib_names"]:
            if isinstance(name, list):
                for i, elem in enumerate(name):
                    if isinstance(elem, list):
                        elem = "-".join([str(int(x)) for x in elem])
                    name[i] = str(elem)
                davids_sys.append("_".join(name))
            else:
                davids_sys.append(name)
        my_sys = systematic_names
        missing = set(davids_sys) - set(my_sys)
        extra = set(my_sys) - set(davids_sys)
        assert not missing, f"Missing systematics compared to David's data: {missing}"
        assert not extra, f"Extra systematics compared to David's data: {extra}"

        # calib = "electron_scattering"
        # david_index = the_data["calib_names"].index(calib)
        # sys_index = systematic_names.index(calib)
        # assert np.allclose(
        #     np.array(systematics)[:, 0, sys_index],
        #     np.array(the_data["d_mBx1c_dcalib_list"])[:, 0, david_index],
        #     atol=1e-6,
        # )

        for davids_index, name in enumerate(the_data["calib_names"]):
            if name in ["lensing_bias"]:  # , "IG_extinction"]:
                continue  # i know its different because I dont do linear interpolation

            for sev in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
                logger.info(f"Checking systematics at sev={sev} for {name}")
                if isinstance(name, list):
                    name = "_".join(name)
                sys_index = systematic_names.index(name)
                for i in range(snia.height):
                    my_systematic = systematics[i][:, sys_index]
                    davids_systematic = np.array(the_data["d_mBx1c_dcalib_list"][i])[:, davids_index]
                    assert np.allclose(
                        my_systematic, davids_systematic, atol=sev
                    ), f"Mismatch in systematic {name} for event {snia['name'][i]}"
                # TODO: lensing bias should be simple and only mB, so use this to check

        # END TODO: REMOVE TEMP CODE

        return result


def _get_redshift_bins(snia: pl.DataFrame, config: Config) -> tuple[pl.DataFrame, RedshiftBins]:
    n_sne = len(snia)
    if snia["z_cmb"].max() == snia["z_cmb"].min() or config.cosmology_model not in [
        CosmologyModel.BINNED_MU,
        CosmologyModel.BINNED_MU_COMOVING_INTERPOLATION,
    ]:
        logger.info("No redshift bins needed for current cosmology model.")
        redshift = snia["z_cmb"].first()

        df = snia.with_columns(mu_const=pl.lit(0), dmu_const_dz=pl.lit(0))
        res: RedshiftBins = {
            "zbins": np.array([redshift]),
            "dmu_dbin": np.ones((n_sne, 1)),
            "dmudz_dbin": np.zeros((n_sne, 1)),
        }
        return df, res

    logger.info(f"Cosmology model {config.cosmology_model} requires redshift bins, generating them now.")
    zsort = np.sort(snia["z_cmb"].to_list())
    zbins = [zsort[-1] * 1.001]
    step = 10
    minstepsize = 0.1
    min_sn_bin = 10
    ind = -1 - min_sn_bin
    z_cutoff_for_05 = 0.8

    while step > minstepsize:
        step = zbins[0] - zsort[ind]
        minstepsize = ((zbins[0] + zsort[ind]) * 0.5 > z_cutoff_for_05) * 0.05 + 0.05

        if step > minstepsize:
            zbins = [zsort[ind]] + zbins
            ind -= min_sn_bin

    logger.info(f"Generated high-z redshift bins: {zbins}")

    zbins = np.concatenate(
        (
            np.linspace(0.05, z_cutoff_for_05, int(0 + np.around(z_cutoff_for_05 / 0.05))),
            np.linspace(z_cutoff_for_05, zbins[0], int(np.around((zbins[0] - z_cutoff_for_05) / 0.1)) + 1)[1:-1],
            zbins,
        )
    )

    logger.info(f"Final redshift bins: {zbins.tolist()}")

    dmu_dbin = np.zeros((n_sne, len(zbins)))
    dmudz_dbin = np.zeros((n_sne, len(zbins)))
    redshifts = np.array(snia["z_cmb"].to_list())
    for j in range(len(zbins)):
        nodes = np.zeros(len(zbins))
        nodes[j] = 1.0

        xvals = np.concatenate(([0.0], zbins))
        yvals = np.concatenate(([-1], nodes))
        ifn = interp1d(xvals, yvals, kind="quadratic")

        dmu_dbin[:, j] = ifn(redshifts)
        dmudz_dbin[:, j] = (ifn(redshifts + 1e-3) - ifn(redshifts)) / 1e-3

    # TODO: get mu

    cosmology = FlatLambdaCDM(H0=70, Om0=0.3)  # type: ignore
    mu_const = cosmology.distmod(redshifts).value  # type: ignore
    dmu_const_dz = 1000 * (cosmology.distmod(redshifts + 0.001).value - mu_const)  # type: ignore
    snia = snia.with_columns(mu_const=pl.Series(mu_const), dmu_const_dz=pl.Series(dmu_const_dz))
    res: RedshiftBins = {
        "zbins": zbins,
        "dmu_dbin": dmu_dbin,
        "dmudz_dbin": dmudz_dbin,
    }
    return snia, res


def _get_bao_cmb_omw0wa(config: Config) -> BAOCMB_Omw0wa:
    if config.fix_omega_m:
        logger.info("Omega_m is fixed, skipping BAO+CMB loading.")
        return {
            "mean": np.array([0.3, -1, 0]),
            "cov": np.diag([100.0, 100.0, 100.0]),
        }
    file = config.data_dir / config.bao_cmb_file
    logger.info(f"Loading BAO+CMB Omw0wa data from {file}.")
    content = json.loads(file.read_text())
    return {
        "mean": np.array(content["mean"]),
        "cov": np.array(content["cov"]),
    }


def _get_redshift_coeffs(snia: pl.DataFrame, config: Config) -> np.ndarray:
    redshifts = np.array(snia["z_cmb"].to_list())
    p_high_mass = np.array(snia["p_high_mass"].to_list())

    n_z = (
        len(config.redshift_coefficient_anchors)
        if config.redshift_coefficient_type == "sample"
        else config.redshift_coefficient_steps
    )
    # This is the number of parameters we need for the latent variables,
    # which doubles if we have separate poluations
    actual_n_x1c_star = n_z
    if config.separate_mass_x1c:
        actual_n_x1c_star *= 2

    redshift_coeffs = np.zeros((len(redshifts), actual_n_x1c_star), dtype=np.float64)

    # If we only have one redshift coefficient, then there's no variation and
    # the coefficients are constant.
    if n_z == 1:
        logger.info("Only one redshift coefficient specified, using constant coefficients.")
        if config.separate_mass_x1c:
            redshift_coeffs[:, 0] = p_high_mass
            redshift_coeffs[:, 1] = 1 - p_high_mass
        else:
            redshift_coeffs[:, :] = 1.0
        return redshift_coeffs

    if config.redshift_coefficient_type == "a":
        logger.info("Using scale-factor based redshift coefficients.")
        a_list = 1.0 / (1.0 + redshifts)
        a_nodes = np.linspace(min(a_list) - 1e-5, max(a_list) + 1e-5, n_z)

        for i in range(len(redshifts)):
            for j in range(n_z):
                coeffs = np.zeros(n_z, dtype=np.float64)
                coeffs[j] = 1

                ifn = interp1d(a_nodes, coeffs, kind="linear")

                if config.separate_mass_x1c:
                    redshift_coeffs[i, j] = ifn(a_list[i]) * p_high_mass[i]
                    redshift_coeffs[i, n_z + j] = ifn(a_list[i]) * (1.0 - p_high_mass[i])
                else:
                    redshift_coeffs[i, j] = ifn(a_list[i])
        return redshift_coeffs

    # Otherwise we're using sample based coefficients
    logger.info("Using sample-based redshift coefficients.")
    zs_to_match = np.array(config.redshift_coefficient_anchors)

    # It seems that the way the sample works is that we use each surveys mean redshift
    # to then we find which anchor it's closest to. For example, if we dont have separate_mass_x1c,
    # and we have three anchors at [0.0, 0.4, 1.0] with 100 SNIa, then we have a (100,3) array
    # and the p_high_mass is put in the column corresponding to the closest anchor.
    samples = (
        snia.group_by("survey")
        .agg(mean_z=pl.col("z_cmb").mean())
        .sort("mean_z")
        .with_columns(
            pl.col("mean_z")
            .map_elements(lambda z: np.argmin(np.abs(float(z) - zs_to_match)), return_dtype=pl.Int16)
            .alias("anchor_index")
        )
    )
    logger.debug(f"Surveys and their mean redshifts: {json.dumps(samples.to_dicts(), indent=4)}")
    snia = snia.join(samples.select(["survey", "anchor_index"]), on="survey", how="left")
    for i, anchor in enumerate(snia["anchor_index"]):
        if config.separate_mass_x1c:
            redshift_coeffs[i, anchor] = p_high_mass[i]
            redshift_coeffs[i, n_z + anchor] = 1 - p_high_mass[i]
        else:
            redshift_coeffs[i, anchor] = 1.0
    return redshift_coeffs


def get_filtered_and_augmented_snia(
    all_supernova: pl.DataFrame,
    config: Config,
    mag_cut_file: pl.DataFrame,
    calibration_uncertainties: dict[str, float],
    calibration_standards: dict[str, Literal["L", "S", "P"]],
) -> pl.DataFrame:
    return (
        impute_snia(all_supernova, config)
        .pipe(pick_source_of_derivatives)
        .pipe(filter_snia, config)
        .pipe(flag_weird_supernova)
        .pipe(add_sample_index)
        .pipe(add_supernova_index)
        .pipe(determine_calibrators, config)
        .pipe(add_lensing_bias, config)
        .pipe(add_prob_high_mass)
        .pipe(add_photoz_errors)
        .join(mag_cut_file, on="survey", how="left")
        .pipe(add_mobs_cuts, config.data_dir)
        .pipe(add_pecv_and_bulk_flow_uncertainties, config)
        .pipe(add_MBEBV_uncertainties, config)
        .pipe(add_intergalactic_extinction_uncertainties, config)
        .pipe(add_electron_scattering_uncertainties, config)
        .pipe(add_landolt_smith_uncertainties, calibration_standards)
        # Now we rescale the uncerts based on the calibration uncertainty scaling factors
        .pipe(rescale_uncertainties, calibration_uncertainties)
        .pipe(remap_x1, config)
        # At this point we should be done with all the deriv_ columns, having
        # processed them into uncertainty_ columns, and can drop them.
        .drop(cs.starts_with("deriv_") | cs.starts_with("instrument|"))
        .sort("name")
    )


def pick_source_of_derivatives(snia: pl.DataFrame) -> pl.DataFrame:
    # In the original read_and_sample_H0.py:50, the derivatives from
    # model_deriv.data were used by default. However, they were marked as
    # failing their convergence check if the "Check" row has abs(log(dx/dP))>0.2
    # for x in [mu, mB, x1, color]. We replicate this logic here, albeit in
    # a vectorised fashion

    # This column is our temp flag signalling we can use the model_deriv columns
    snia = snia.with_columns(
        using_model_derivs=pl.all_horizontal(
            pl.col("model_deriv_Check_All_dmu/dP").log().abs().le(0.2),
            pl.col("model_deriv_Check_All_dmB/dP").log().abs().le(0.2),
            pl.col("model_deriv_Check_All_ds/dP").abs().le(0.2),
            pl.col("model_deriv_Check_All_dc/dP").abs().le(0.2),
        )
    )
    model_snia = (
        snia.filter(pl.col("using_model_derivs"))
        .drop(cs.starts_with("result_deriv_"))
        .rename(lambda c: c.removeprefix("model_"))
    )
    result_snia = (
        snia.filter(~pl.col("using_model_derivs") | pl.col("using_model_derivs").is_null())
        .drop(cs.starts_with("model_deriv_"))
        .rename(lambda c: c.removeprefix("result_"))
    )
    return pl.concat([model_snia, result_snia], how="diagonal_relaxed")


def add_supernova_index(snia: pl.DataFrame) -> pl.DataFrame:
    """Adds a supernova_index column to the supernova dataframe, which is an integer index for each supernova."""
    logger.info(f"Adding supernova index for {snia.height} supernova.")
    return snia.with_row_index("supernova_index")


def add_sample_index(snia: pl.DataFrame) -> pl.DataFrame:
    """Adds a sample_index column to the supernova dataframe, which is an integer index for each unique survey."""
    samples = snia.select("survey").unique().sort("survey").with_row_index("sample_index")
    logger.info(f"Identified {samples.height} unique supernova samples/surveys. Adding their index.")
    return snia.join(samples, on="survey", how="left")


def add_prob_high_mass(snia: pl.DataFrame) -> pl.DataFrame:
    """Adds the probability that each supernova is in a high mass host galaxy."""
    p_high_mass = 0.5 * (1.0 + erf((pl.col("mass") - 10.0) / (np.sqrt(2.0) * pl.col("mass_err"))))
    return snia.with_columns(p_high_mass=p_high_mass)


def condense_systematics(snia: pl.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    """Condenses the uncertainty_ columns so each row (supernova) has a new column d_mBx1c_d_calib which has a shape of (3, n_calib)
    where n_calib is the number of calibration systematics we have. This is needed for Stan to process the data correctly."""
    calib = []
    cols = sorted(list(set([col.split("_", maxsplit=2)[-1] for col in snia.columns if col.startswith("uncertainty_")])))
    num = len(cols)
    for row in snia.iter_rows(named=True):
        matrix = np.zeros((3, num), dtype=np.float64)
        for i, col in enumerate(cols):
            matrix[0, i] = row.get(f"uncertainty_mB_{col}", 0.0) or 0.0
            matrix[1, i] = row.get(f"uncertainty_x1_{col}", 0.0) or 0.0
            matrix[2, i] = row.get(f"uncertainty_color_{col}", 0.0) or 0.0
        calib.append(matrix)
    logger.info(f"Condensed systematics into {num} calibration terms for {len(calib)} supernova.")
    logger.debug(f"Calibration terms are: {json.dumps(cols, indent=2)}")
    return cols, calib


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
            cov_mBx1=pl.col("cov_mBx1") * pl.col("new_x1_slope"),
            cov_x1c=pl.col("cov_x1c") * pl.col("new_x1_slope"),
            cov_x1x1=pl.col("cov_x1x1") * pl.col("new_x1_slope") ** 2,
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


def load_calibration_uncertainties(config: Config) -> tuple[dict[str, float], dict[str, Literal["L", "S", "P"]]]:
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

    # We also want to return the calibration system, which is a dict from the name to L, S, P
    calibs_with_standard = calib_df.filter(pl.col("standard").is_not_null())
    standards = dict(zip(calibs_with_standard["key"], calibs_with_standard["standard"]))
    return combined, standards


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
        snia = snia.with_columns(uncertainty_mB_lensing_bias=0.5 * (config.lensing_dispersion * pl.col("z_cmb")) ** 2)

    return snia


def add_photoz_errors(snia: pl.DataFrame) -> pl.DataFrame:
    if "photo_z0" not in snia.columns:
        logger.info("No photo-z supernova found, skipping photo-z error addition.")
        return snia.with_columns(
            photo_z0=pl.lit(None),
            photo_sigz=pl.lit(None),
            photo_spikez=pl.lit(None),
            photoz_index=pl.lit(0),
            photo_pspike=pl.lit(None),
        )

    logger.info(
        f"Adding photo-z error terms for {snia.filter(pl.col("photo_z0").is_not_null()).height} photo-z supernova."
    )
    return snia.with_columns(
        dz_uncertainty_photoz_mB=pl.when(pl.col("photoz_mean").is_not_null())
        .then(pl.col("deriv_Redshift_dmB/dP"))
        .otherwise(None),
        dz_uncertainty_photoz_x1=pl.when(pl.col("photoz_mean").is_not_null())
        .then(pl.col("deriv_Redshift_ds/dP"))
        .otherwise(None),
        dz_uncertainty_photoz_c=pl.when(pl.col("photoz_mean").is_not_null())
        .then(pl.col("deriv_Redshift_dc/dP"))
        .otherwise(None),
        photoz_index=pl.when(pl.col("photoz_mean").is_not_null()).then(pl.col("supernova_index")).otherwise(0),
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
            (
                config.peculiar_velocity_dispersion
                * (5 / np.log(10))
                * (pl.col("z_cmb") + 1)
                / (pl.col("z_cmb") * (pl.col("z_cmb") * 0.5 + 1))
            )
            ** 2
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
            total_bulk_quad = float(np.sum(eigenvector**2))
            total_pec_v_on_diag = float(np.clip(row["pec_vel_on_diag"] - total_bulk_quad, 0, 100))
            final_pecvs[name] = total_pec_v_on_diag
            logger.debug(f"SN {name}: {total_bulk_quad=}, {total_pec_v_on_diag=}")

    # reclip pec_vel_on_diag to between 0 and 100
    snia = snia.with_columns(pl.col("pec_vel_on_diag").clip(0, 100))

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
    ).with_columns(cov_mBmB=pl.col("cov_mBmB") + pl.col("pec_vel_on_diag"))

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


def _get_redshifts(redshifts: list[float]) -> RedshiftSimps:
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
            "covrestframemag_0_bx1": "cov_mBx1",
            "covcolorrestframemag_0_b": "cov_mBc",
            "covx1x1": "cov_x1x1",
            "covcolorx1": "cov_x1c",
            "covcolorcolor": "cov_cc",
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


def filter_snia(df: pl.DataFrame, config: Config):
    if config.weird_sn_file is None:
        weird_sn = []
        logger.info("No weird SN file provided, skipping weird SN exclusion.")
    else:
        weird_sn_file = config.data_dir / config.weird_sn_file
        weird_sn = [str(x) for x in yaml.safe_load(weird_sn_file.read_text())["weird_sn_names"]]
        logger.info(f"Loaded {len(weird_sn)} weird SN names from {weird_sn_file}.")

    df_filtered = df.filter(
        pl.col("lcfit_passed")
        & pl.col("z_cmb").is_between(config.filters.min_redshift, config.filters.max_redshift)
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
    # Drop columns which are all null after filtering
    df_filtered = df_filtered[[s.name for s in df_filtered if not (s.null_count() == df_filtered.height)]]
    logger.info(f"Filtered to {df_filtered.height} SNe Ia")
    return df_filtered.sort("name")


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
            .then(pl.when(pl.col("z_cmb") > 0.1).then(10.0).otherwise(11.0))
            .otherwise(pl.col("mass")),
            mass_err=pl.when(pl.col("bad_mass")).then(1).otherwise(pl.col("mass_err")),
        )
        # Add in cov_mBmB from mb_err and the lensing dispersion
        .with_columns(cov_mBmB=pl.col("mB_err") ** 2 + (pl.col("z_cmb") * config.lensing_dispersion) ** 2)
        .drop("mB_err")  # Drop the mB_err column so it cant be used instead of cov_mBmB
        # And to make my life easier, I'm going to rename the supernova to put their survey in the name
        .with_columns(
            original_name=pl.col("name"),
            name=pl.col("survey") + "_" + pl.col("name"),
            in_cluster=pl.col("in_cluster").cast(pl.Int8),
        )
    )
