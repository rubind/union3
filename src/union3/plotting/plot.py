import matplotlib.pyplot as plt
import polars as pl

from union3.config import Config
from union3 import logger
import numpy as np


def plot_approx_hubble_diagram(
    data: pl.DataFrame,
    config: Config,
    MB: float = -19.3,
    alpha: float = 0.14,
    beta: float = 3.1,
):
    if not config.do_plotting:
        return
    fig, ax = plt.subplots()
    ax.errorbar(
        data["redshift"],
        data["mb"] - MB,
        yerr=np.sqrt(data["mb_err"] ** 2 + (alpha * data["x1_err"]) ** 2 + (beta * data["color_err"]) ** 2),
        fmt="o",
        markersize=4,
        alpha=0.5,
    )
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Distance Modulus")
    ax.set_title("Hubble Diagram")
    output_path = config.output_dir / "hubble_diagram.webp"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Hubble diagram to {output_path}")
