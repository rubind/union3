from union3 import logger
from union3 import Config, Data, Model
# from union3.plotting import plot_approx_hubble_diagram


def main():
    config = Config()
    logger.info(f"Running Unity with base config file: {config.base}")
    logger.info(f"Run settings: {config.model_dump_json(indent=2)}")

    data = Data.from_config(config)

    model = Model.from_config(config)
    model.initialise(data)

    samples = model.fit()
    samples.write_parquet(config.output_dir / "mcmc_samples.parquet")

    print(samples.describe())
    # plot_approx_hubble_diagram(data.filtered_supernova, config)


if __name__ == "__main__":
    main()
