import polars as pl
from loguru import logger
from union3.config import Config
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import numpy as np


def add_MBEBV_uncertainties(snia: pl.DataFrame, config: Config) -> pl.DataFrame:
    sig_stat = 0.16  # 16% statistical uncertainty
    sig_norm = 0.10  # 10% multiplicative normalization uncertainty
    sig_add = config.MWEBV_zeropoint_EBV  # E.g., 5 mmag E(B-V) additive uncertainty

    snia = snia.with_columns(
        uncertainty_mB_MWEBV_multnorm=pl.col("deriv_MWEBV_dmB/dP") * pl.col("MWEBV") * sig_norm * sig_stat,
        uncertainty_x1_MWEBV_multnorm=pl.col("deriv_MWEBV_ds/dP") * pl.col("MWEBV") * sig_norm * sig_stat,
        uncertainty_color_MWEBV_multnorm=pl.col("deriv_MWEBV_dc/dP") * pl.col("MWEBV") * sig_norm * sig_stat,
        uncertainty_mB_MWEBV_addnorm=pl.col("deriv_MWEBV_dmB/dP") * sig_add,
        uncertainty_x1_MWEBV_addnorm=pl.col("deriv_MWEBV_ds/dP") * sig_add,
        uncertainty_color_MWEBV_addnorm=pl.col("deriv_MWEBV_dc/dP") * sig_add,
        extra_cov_mBmB_MWEBV=(pl.col("deriv_MWEBV_dmB/dP") * sig_stat * pl.col("MWEBV")) ** 2,
        extra_cov_x1x1_MWEBV=(pl.col("deriv_MWEBV_ds/dP") * sig_stat * pl.col("MWEBV")) ** 2,
        extra_cov_cc_MWEBV=(pl.col("deriv_MWEBV_dc/dP") * sig_stat * pl.col("MWEBV")) ** 2,
        extra_cov_mBx1_MWEBV=(pl.col("deriv_MWEBV_dmB/dP") * pl.col("deriv_MWEBV_ds/dP"))
        * (sig_stat * pl.col("MWEBV")) ** 2,
        extra_cov_mBc_MWEBV=(pl.col("deriv_MWEBV_dmB/dP") * pl.col("deriv_MWEBV_dc/dP"))
        * (sig_stat * pl.col("MWEBV")) ** 2,
        extra_cov_x1c_MWEBV=(pl.col("deriv_MWEBV_ds/dP") * pl.col("deriv_MWEBV_dc/dP"))
        * (sig_stat * pl.col("MWEBV")) ** 2,
    )

    return snia


def add_intergalactic_extinction_uncertainties(snia: pl.DataFrame, config: Config) -> pl.DataFrame:
    if config.intergalactic_extinction_coefficient == 0.0:
        logger.info("Skipping addition of intergalactic extinction uncertainties.")
        return snia

    assert config.intergalactic_extinction_file is not None, "Intergalactic extinction file must be specified."
    data_file = config.data_dir / config.intergalactic_extinction_file
    logger.info(f"Adding intergalactic extinction uncertainties from {data_file}.")

    with fits.open(data_file) as hdul:
        data = hdul[0].data  # type: ignore

    interpolation = RectBivariateSpline(data[1:, 0], data[0, 1:], data[1:, 1:], kx=1, ky=1)

    def interp(redshift: float, wavelength_angrstroms: float) -> float:
        effective_wavelength = wavelength_angrstroms / 10000.0  # Convert to microns
        return float(interpolation(redshift, effective_wavelength)[0, 0])

    # The original get_IG_extinction_sys has a few checks here I blindly replicate
    # TODO: ask David about these so I can put proper comments in
    # The further away you are, the more dust you've travelled through
    assert interp(1.0, 4400) > interp(0.5, 4400), "Extinction at z=1.0 should be greater than at z=0.5."
    # Red wavelengths are more effected than blue ones
    assert interp(1.0, 4400) > interp(1.0, 5500), "Extinction at 4400A should be greater than at 5500A."

    scale = config.intergalactic_extinction_coefficient

    # Each supernova has Zeropoint calibration uncertainty that give gradients
    # in mB, x1, and color. Each filter is parametrised by an effective rest wavelength.
    # We want to use this wavelength to determine the scale of intergalactic extinction uncertainty.
    # And then scale the gradients in mB, x1, and color by this uncertainty, before adding it to our
    # dataframe under an "IG_extinction" key, as this key is what the uncertainty rescaling function expects.

    # First, let us compute all the scales as a lookup from name to magsys/instrument/band
    long_data = (
        snia.unpivot(index=["name", "z_cmb"])
        .filter(pl.col("variable").str.starts_with("deriv_Zeropoint"))
        .filter(pl.col("value").is_not_null())
        .with_columns(pl.col("value").cast(pl.Float64))
    )
    redshifts = long_data.filter(pl.col("variable").str.ends_with("RestLamb"))

    # Add in the scaling factor
    scale = redshifts.with_columns(
        scale=pl.struct("z_cmb", "value").map_elements(lambda x: interp(x["z_cmb"], x["value"]), pl.Float64) * scale
    ).select("name", "scale")
    # And now we need to turn these into IG_extinction uncertainties, which means columns that look like
    # uncertainty_mB_IG_extinction, uncertainty_x1_IG_extinction, uncertainty_color_IG_extinction
    ucertainties = (
        long_data.filter(pl.col("variable").str.contains("/dP"))
        .join(scale, on="name")
        .with_columns(value=pl.col("value") * pl.col("scale"))
        .group_by("name")
        .agg(
            uncertainty_mB_IG_extinction=pl.col("value").filter(pl.col("variable").str.contains("dmB/dP")).sum(),
            uncertainty_x1_IG_extinction=pl.col("value").filter(pl.col("variable").str.contains("ds/dP")).sum(),
            uncertainty_color_IG_extinction=pl.col("value").filter(pl.col("variable").str.contains("dc/dP")).sum(),
        )
        .sort("name")
    )
    return snia.join(ucertainties, on="name", how="left")


def add_electron_scattering_uncertainties(snia: pl.DataFrame, config: Config) -> pl.DataFrame:
    tau = config.electron_scattering_tau
    dtau = config.electron_scattering_dtau
    z = pl.col("z_cmb")

    snia = (
        snia.with_columns(
            tau_electron=(tau * ((0.3 * (1 + z) ** 3 + 0.7).sqrt() - 1)),
            dtau_electron=(dtau * ((0.3 * (1 + z) ** 3 + 0.7).sqrt() - 1)),
        )
        .with_columns(
            add_mag_electron=-2.5 * pl.col("tau_electron") / np.log(10),
            uncertainty_mB_electron_scattering=2.5 * pl.col("dtau_electron") / np.log(10),
        )
        .with_columns(mB=pl.col("mB") + pl.col("add_mag_electron"))
    )

    assert all(snia["add_mag_electron"] < 0), "Electron scattering magnitudes should be negative (dimming)."

    return snia


def rescale_uncertainties(snia: pl.DataFrame, calibration_uncertainties: dict[str, float]) -> pl.DataFrame:
    logger.info("Rescaling uncertainties based on calibration uncertainties.")

    # What we want to do here is loop over each column prefixed with uncertainty_
    # And try to lookup the corresponding calibration uncertainty from the calibration_uncertainties dict
    # For example, a uncertainty_mB_doom would hopefully mean there is a 'doom' key in
    # the calibration_uncertainties dict.
    cols = [col for col in snia.columns if col.startswith("uncertainty_")]
    expressions = []
    for col in cols:
        _, _, key = col.split("_", maxsplit=2)
        scaling_factor = calibration_uncertainties.get(key)
        if scaling_factor is None:
            logger.warning(f"No calibration uncertainty found for key {key}, skipping rescaling for {col}.")
            continue
        logger.info(f"Rescaling {col} by calibration uncertainty {scaling_factor} for key {key}.")
        expressions.append(pl.col(col) * scaling_factor)

    return snia.with_columns(expressions)
