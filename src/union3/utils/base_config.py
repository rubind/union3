from pathlib import Path
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    SecretsSettingsSource,
    SettingsConfigDict,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)


class DeferredYamlConfigSettingsSource(YamlConfigSettingsSource):
    """YAML config settings source that can reload config files later."""

    def __call__(self):
        """Reload config files based on current state."""
        yaml_file_to_reload = self.current_state.get("base", self.yaml_file_path)
        if yaml_file_to_reload is not None:
            config_dir = Path(__file__).parent.parent / "configs"
            names = [p.name for p in config_dir.glob("*.yml")]
            assert (
                yaml_file_to_reload in names
            ), f"Config file {yaml_file_to_reload} not found in configs directory, options are {names}"
            yaml_file_to_reload = config_dir / yaml_file_to_reload

        super().__init__(
            settings_cls=self.settings_cls,
            yaml_file=yaml_file_to_reload,
        )
        return super().__call__()


class FileConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, env_file_encoding="utf-8", env_nested_delimiter="__")

    @classmethod
    def settings_customise_sources(  # type: ignore
        cls,
        settings_cls: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            DeferredYamlConfigSettingsSource(settings_cls),
        )
