from union3 import logger
from union3.config import Config
from union3.data import load_snia_lightcurve_fits


def main():
    config = Config()
    logger.info(f"Running Unity with base config file: {config.base}")
    logger.info(f"Run settings: {config.model_dump_json(indent=2)}")

    full_data = load_snia_lightcurve_fits(config)  # noqa: F841


if __name__ == "__main__":
    main()
