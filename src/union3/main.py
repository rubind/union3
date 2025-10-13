from union3 import logger
from union3.config import Config


def main():
    config = Config()
    logger.info(f"Running Unity with base config file: {config.base}")
    logger.info(f"Run settings: {config.model_dump_json(indent=2)}")


if __name__ == "__main__":
    main()
