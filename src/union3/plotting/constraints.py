import matplotlib.pyplot as plt
import polars as pl
from chainconsumer import ChainConsumer, Chain

from union3.config import Config, CosmologyModel

main_cosmo_parameters = {
    CosmologyModel.OM: ["Om"],
    CosmologyModel.OM_W: ["Om", "wDE"],
    CosmologyModel.Q0_J0: ["q0", "j0"],
    CosmologyModel.OM_W0_WA: ["Om", "wDE", "waDE"],
}


def plot_cosmology_constraints(config: Config, samples: pl.DataFrame) -> None:
    params = main_cosmo_parameters[config.cosmology_model]

    subset = samples.select(params).to_pandas()

    c = ChainConsumer()
    c.add_chain(Chain(samples=subset, name="SNIa constraints"))
    fig = c.plotter.plot()

    output_path = config.output_dir / "constraints.webp"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
