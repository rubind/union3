import matplotlib.pyplot as plt
import polars as pl
import numpy as np

from unity.config import Config
from unity import logger
from unity.data import Data
from chainconsumer.color_finder import colors
from astropy.cosmology import FlatLambdaCDM
from unity import Model
def plot_approx_hubble_diagram(
    data: Data,
    config: Config,
    MB: float = -19.1,
    alpha: float = 0.14,
    beta: float = 3.1,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    color_iterator = colors.next_colour()

    cosmology = FlatLambdaCDM(H0=70, Om0=0.3)  # type: ignore
    surveys = sorted(data.filtered_supernova["survey"].unique().to_list())

    for survey in surveys:
        df = data.filtered_supernova.filter(pl.col("survey") == survey)

        redshift = df["z_cmb"]
        cosmo_mu = cosmology.distmod(redshift.to_numpy()).value  # type: ignore
        distmod = df["mB"] - MB + alpha * df["x1"] - beta * df["color"] - cosmo_mu
        distmod_err = (df["cov_mBmB"] + alpha**2 * df["cov_x1x1"] + beta**2 * df["cov_cc"]).sqrt()
        color = next(color_iterator)
        ax.errorbar(
            redshift,
            distmod,
            yerr=distmod_err,
            fmt="o",
            markersize=0.1,
            elinewidth=0.5,
            alpha=0.15,
            color=color,
        )
        ax.scatter(redshift, distmod, s=1, label=survey, color=color)

    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Distance Modulus Delta to Om=0.3, H0=70")
    ax.set_ylim(-2, 2)
    ax.legend(
        title="Survey",
        fontsize="small",
        title_fontsize="medium",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )
    output_path = config.output_dir / "subtracted_hubble_diagram.webp"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Hubble diagram to {output_path}")


def plot_hubble_diagram_from_stanInputData(model: Model,
    config: Config,
    MB: float = -19.1,
    alpha: float = 0.14,
    beta: float = 3.1,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    color_iterator = colors.next_colour()

    cosmology = FlatLambdaCDM(H0=70, Om0=0.3)  # type: ignore

    # Extract plotting values from the Model.data container
    mBx1c_list = model.data['obs_mBx1c']
    mBx1c_cov_list = model.data['obs_mBx1c_cov']
    sample_list = model.data["sample_list"]
    redshifts = model.data['redshifts']
    has_distmod = model.data['has_distmod']

    # Unpack data and diag covariance
    mB, x1, c = np.array([[mbx1c[i] for i in range(3)] for mbx1c in mBx1c_list]).T
    mB_var, x1_var, c_var = np.array([[mbx1c_cov[i][i] for i in range(3)] for mbx1c_cov in mBx1c_cov_list]).T
    #surveys_unique = surveys = (surveys.unique().to_list())

    # Compute all y-axis values
    cosmo_mu = cosmology.distmod(redshifts).value  # type: ignore
    distmod = mB - MB + alpha * x1 - beta * c - cosmo_mu
    distunc = np.sqrt(mB_var + alpha**2 * x1_var + beta**2 * c_var)

    # Zero to the cosmo fiducial
    distmod -= np.median(distmod)

    # Map Stan's 1-indexed sample_list back to survey names (sample_index is 0-indexed).
    sample_names = dict(model._raw_data.samples.select(["sample_index", "survey"]).iter_rows())

    for sample_index in sorted(sample_names):
        # sample_list is 1-indexed (sample_index + 1; see StanModel.initialise), so match on +1.
        sel = sample_list == sample_index + 1
        flow_inds = np.where(sel & (has_distmod == 0))
        calib_inds = np.where(sel & (has_distmod == 1))

        distmod_calib_i, distmod_flow_i = distmod[calib_inds], distmod[flow_inds]
        distunc_calib_i, distunc_flow_i = distunc[calib_inds], distunc[flow_inds]
        zcalib_i, zflow_i = redshifts[calib_inds], redshifts[flow_inds]
        color = next(color_iterator)
        ax.errorbar(
            zflow_i,
            distmod_flow_i,
            yerr=distunc_flow_i,
            fmt="o",
            markersize=0.1,
            elinewidth=0.5,
            alpha=0.15,
            color=color,
        )

        # Plot calibrators. Kept cosmo-referenced on purpose: their low-z peculiar-velocity
        # scatter is a useful extra layer of blinding, so we do NOT re-reference them to
        # their external distance modulus.
        ax.errorbar(
            zcalib_i,
            distmod_calib_i,
            yerr=distunc_calib_i,
            fmt="*",
            markersize=20,
            elinewidth=1.0,
            alpha=0.9,
            mfc=color,
            mec='black',
        )

        ax.scatter(zflow_i, distmod_flow_i, s=1, label=sample_names[sample_index], color=color)

    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Distance Modulus Delta to Om=0.3, H0=70")
    ax.set_ylim(-2, 2)
    ax.legend(
        title="Survey",
        fontsize="small",
        title_fontsize="medium",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )
    output_path = config.output_dir / "subtracted_hubble_diagram.webp"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Hubble diagram to {output_path}")

