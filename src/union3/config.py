from enum import StrEnum
from pathlib import Path
from typing import Literal, Self
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from union3.utils.base_config import FileConfig


class CosmologyModel(StrEnum):
    OM = "om"
    BINNED_MU = "binned_mu"
    OM_W = "om_w"
    Q0_J0 = "q0_j0"
    OM_W0_WA = "om_w0_wa"
    BINNED_MU_COMOVING_INTERPOLATION = "binned_mu_comoving_interpolation"


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
        assert (
            self.min_redshift < self.max_redshift
        ), f"min_redshift ({self.min_redshift}) must be less than max_redshift ({self.max_redshift})"
        assert (
            self.min_color < self.max_color
        ), f"min_color ({self.min_color}) must be less than max_color ({self.max_color})"
        return self


class Config(FileConfig):
    base: str | None = Field(
        default=None,
        description="Base config YAML file in the configs directory. If None, will use the defaults from config.py",
    )
    data_dir: Path = Field(default=Path(__file__).parent / "data/resources")
    output_dir: Path = Field(default=Path(__file__).parents[2] / "output")

    filters: FilterConfig = Field(default_factory=FilterConfig)

    #! Other file inputs
    mag_cut_file: str = Field(
        default="mapping/mag_cut.csv", description="Mag cut mapping file relative to data directory."
    )
    distance_ladder_file: str | None = Field(
        default="distance_ladder/dist_ladder_R22.csv",
        description="Distance ladder file relative to data directory, used to determine calibrators.",
        examples=["distance_ladder/dist_ladder_R22.csv"],
    )
    lensing_bias_file: str = Field(
        default="lensing/lensing_bias.csv",
        description="Lensing bias file relative to data directory.",
    )
    calibration_uncertainties_file: str = Field(
        default="calibration/calibration_uncertainties.csv",
        description="Calibration uncertainties file relative to data directory.",
    )
    weird_sn_file: str | None = Field(
        default="misc/weird_sn.yml",
        description="File with list of weird SN names to exclude, relative to data directory.",
    )
    intergalactic_extinction_file: str | None = Field(
        default="extinction/Azwave_grid.fits",
        description="File with intergalactic extinction data, relative to data directory.",
    )
    bao_cmb_file: str = Field(
        default="other_cosmology/BAOCMB_Omw0wa.json",
        description="File with BAO+CMB constraints, relative to data directory.",
    )

    #! Config to control what gets run
    cache_model_fitting: bool = Field(default=True, description="Use caching for stan model fitting if possible.")
    do_plotting: bool = Field(default=True, description="Generate plots after running.")

    #! Cosmology model config
    cosmology_model: CosmologyModel = Field(
        default=CosmologyModel.OM_W0_WA, description="Cosmology model to use for fitting."
    )
    fit_model: str = Field(default="unity_1.8.stan", description="Stan model file in the models directory.")
    iterations: int = Field(default=20, ge=1, description="Number of iterations for MCMC.")
    warmup_iterations: int = Field(default=10, ge=1, description="Number of warmup iterations for MCMC.")
    refresh_iterations: int = Field(
        default=5, ge=0, description="Number of iterations between progress updates for MCMC. 0 to turn off."
    )
    num_chains: int = Field(default=4, ge=1, description="Number of chains for MCMC.")
    extra_single_dimension_parameters_only: bool = Field(
        default=True, description="Whether to only save extra single-dimension parameters."
    )
    max_params_to_save: int = Field(default=1000, ge=1, description="Maximum number of parameters to save from MCMC.")
    do_host_mass: bool = Field(default=True, description="Whether to include host mass step correction.")
    fix_omega_m: bool = Field(default=False, description="Whether to fix Omega_m during fitting to 0.3.")
    MB_by_sample: bool = Field(default=False, description="Whether to fit for different absolute magnitude by sample.")
    include_peculiar_velocity_covariance: bool = Field(
        default=True, description="Whether to include peculiar velocity covariance matrix."
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
    use_lensing_file: bool = Field(
        default=False,
        description="Whether to use lensing bias file to add lensing uncertainties. If false, use lensing_dispersion.",
    )
    lensing_dispersion: float = Field(
        default=0.055, ge=0.0, description="Lensing dispersion in magnitudes per redshift."
    )
    MWEBV_zeropoint_EBV: float = Field(default=0.005, ge=0.0, description="Zeropoint uncertainty on Milky Way E(B-V).")
    outlier_fraction: float = Field(default=0.02, ge=0.0, le=1.0, description="Fraction of outliers in the sample.")
    intergalactic_extinction_coefficient: float = Field(
        default=1.0, description="The scale factor to apply to intergalactic extinction uncertainties."
    )
    # TODO: ask david for better descriptions of these
    electron_scattering_tau: float = Field(default=0.0042, description="Electron scattering?")
    electron_scattering_dtau: float = Field(default=0.00042, ge=0.0, description="Electron scattering grad?")
    remap_x1_intercept: float = Field(default=0.0, description="Remapping to apply to x1.")
    remap_x1_slope: float = Field(default=0.0, description="Remapping slope to apply to x1.")

    redshift_coefficient_type: Literal["sample", "a"] = Field(
        default="sample", description="Type of redshift coefficient to use, either 'sample' or 'a' (scale factor)."
    )
    redshift_coefficient_anchors: list[float] = Field(
        default=[0.0, 0.4, 1.0],
        min_length=1,
        description="Redshift anchors for redshift coefficients when using 'sample' type.",
    )
    redshift_coefficient_steps: int = Field(
        default=1, ge=1, description="Number of steps for redshift coefficients when using 'a' type."
    )
    threeD_unexplained: bool = Field(default=False, description="TODO: ask david")
    do_two_alpha_beta: bool = Field(default=False, description="Whether to fit for two alpha and beta values.")

    @property
    def model_path(self) -> Path:
        return self.model_dir() / self.fit_model

    @classmethod
    def model_dir(cls):
        return Path(__file__).parent / "models"

    @field_validator("fit_model")
    def validate_stan_model(cls, v: str) -> str:
        names = [p.name for p in cls.model_dir().glob("*")]
        assert v in names, f"Model {v} not found in models directory, options are {names}"
        return v

    @field_validator("electron_scattering_tau")
    def validate_electron_scattering_tau(cls, v: float) -> float:
        assert abs(v) < 0.1, "electron_scattering_tau should be less than 0.1"
        assert v > -0.001, "electron_scattering_tau should be greater than -0.001"
        return v

    @field_validator("electron_scattering_dtau")
    def validate_electron_scattering_dtau(cls, v: float) -> float:
        assert abs(v) < 0.01, "electron_scattering_dtau should be less than 0.01"
        return v

    def _check_file_exists(self, data_dir: Path, file_name: str | None) -> None:
        if file_name is None:
            return
        file_path = data_dir / file_name
        assert file_path.exists(), f"File {file_name} does not exist in data directory {data_dir}."

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        assert self.data_dir.exists(), f"Data directory {self.data_dir} does not exist."
        self._check_file_exists(self.data_dir, self.mag_cut_file)
        self._check_file_exists(self.data_dir, self.distance_ladder_file)
        self._check_file_exists(self.data_dir, self.weird_sn_file)
        self._check_file_exists(self.data_dir, self.lensing_bias_file)
        self._check_file_exists(self.data_dir, self.calibration_uncertainties_file)
        self._check_file_exists(self.data_dir, self.intergalactic_extinction_file)
        self._check_file_exists(self.data_dir, self.bao_cmb_file)

        assert (
            self.cosmology_model != CosmologyModel.BINNED_MU_COMOVING_INTERPOLATION
        ), "BINNED_MU_COMOVING_INTERPOLATION is deprecated for now."
        return self
