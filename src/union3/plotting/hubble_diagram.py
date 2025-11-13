import matplotlib.pyplot as plt
import polars as pl

from union3.config import Config
from union3 import logger
from union3.data import Data
from chainconsumer.color_finder import colors
from astropy.cosmology import FlatLambdaCDM


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
