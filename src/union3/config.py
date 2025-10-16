from pathlib import Path
from typing import Self
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from union3.utils.base_config import FileConfig


class FilterConfig(BaseSettings):
    min_redshift: float = Field(default=0.01, ge=0.0, description="Cut on minimum redshift.")
    max_redshift: float = Field(default=3.0, ge=0.0, description="Cut on maximum redshift.")
    max_first_phase: float = Field(default=100.0, description="Cut on maximum first phase.")
    min_last_phase: float = Field(default=-100.0, description="Cut on minimum last phase.")
    max_color_uncertainty: float = Field(default=0.2, ge=0.0, description="Cut on maximum color uncertainty.")
    min_color: float = Field(default=-0.3, description="Cut on minimum color.")
    max_color: float = Field(default=0.3, description="Cut on maximum color.")
    max_MWEBV: float = Field(default=0.3, description="Cut on maximum Milky Way E(B-V).")

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        assert self.min_redshift < self.max_redshift, "min_redshift must be less than max_redshift"
        return self


class Config(FileConfig):
    base: str | None = Field(
        default=None,
        description="Base config YAML file in the configs directory. If None, will use the defaults from config.py",
    )
    data_dir: Path = Field(default=Path(__file__).parents[2] / "data")
    output_dir: Path = Field(default=Path(__file__).parents[2] / "output")

    filters: FilterConfig = Field(default_factory=FilterConfig)

    #! Config to control what gets run
    cache_data_processing: bool = Field(
        default=True, description="Use caching for data processing and stan running if possible."
    )
    cache_model_fitting: bool = Field(default=True, description="Use caching for stan model fitting if possible.")
    do_plotting: bool = Field(default=True, description="Generate plots after running.")

    #! Cosmology model config
    model: str = Field(default="unity_stan_1.8.txt", description="Stan model file in the models directory.")
    iterations: int = Field(default=2500, ge=1, description="Number of iterations for MCMC.")
    num_jobs: int = Field(default=4, ge=1, description="Number of parallel jobs for MCMC.")
    num_chains: int = Field(default=4, ge=1, description="Number of chains for MCMC.")
    max_params_to_save: int = Field(default=1000, ge=1, description="Maximum number of parameters to save from MCMC.")
    do_host_mass: bool = Field(default=True, description="Whether to include host mass step correction.")
    fix_omega_m: bool = Field(default=False, description="Whether to fix Omega_m during fitting to 0.3.")
    MB_by_sample: bool = Field(default=False, description="Whether to fit for different absolute magnitude by sample.")
    include_peculiar_velocity_covariance: bool = Field(
        default=False, description="Whether to include peculiar velocity covariance matrix."
    )
    separate_mass_x1c: bool = Field(
        default=True, description="Whether to separate x1 and color standardization by host mass."
    )

    #! Data processing config
    do_blinding: bool = Field(default=True, description="Whether to blind the data.")

    #! Data augmentation config
    peculiar_velocity_dispersion: float = Field(
        default=0.001, ge=0.0, description="Peculiar velocity dispersion in units of c."
    )
    lensing_dispersion: float = Field(
        default=0.055, ge=0.0, description="Lensing dispersion in magnitudes per redshift."
    )
    MWEBV_zeropoint_EBV: float = Field(default=0.005, ge=0.0, description="Zeropoint uncertainty on Milky Way E(B-V).")
    outlier_fraction: float = Field(default=0.02, ge=0.0, le=1.0, description="Fraction of outliers in the sample.")
    # blinding_file?
    # filename_list
    # weird_sn_list
    # mag_cuts
    # sample file
    # calibration_uncertainties
    # max params to save

    @property
    def model_path(self) -> Path:
        return self.model_dir() / self.model

    @classmethod
    def model_dir(cls):
        return Path(__file__).parent / "models"

    @field_validator("model")
    def validate_stan_model(cls, v: str) -> str:
        names = [p.name for p in cls.model_dir().glob("*.txt")]
        assert v in names, f"Model {v} not found in models directory, options are {names}"
        return v

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self
