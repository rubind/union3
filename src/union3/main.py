from union3 import logger
from union3.config import Config
from union3.data import Data
from union3.plotting import plot_approx_hubble_diagram


def main():
    config = Config()
    logger.info(f"Running Unity with base config file: {config.base}")
    logger.info(f"Run settings: {config.model_dump_json(indent=2)}")

    data = Data.from_config(config)

    plot_approx_hubble_diagram(data.filtered_supernova, config)


if __name__ == "__main__":
    main()
