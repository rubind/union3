from pathlib import Path
import polars as pl
import numpy as np
from union3 import Config, Data, logger


class Model:
    def initialise(self, data: Data) -> None:
        raise NotImplementedError()

    def fit(self) -> pl.DataFrame:
        """Fit the model as per the data and config, returning the fit chains as a DataFrame."""
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: Config) -> "Model":
        model_path = config.model_path
        if model_path.suffix == ".stan":
            return StanModel(model_path, config)
        else:
            raise ValueError(f"Unsupported model file extension: {model_path.suffix}")


class StanModel(Model):
    def __init__(self, model_path: Path, config: Config):
        assert model_path.exists(), f"Model file {model_path} does not exist."
        self.model_path = model_path
        self.model_text = self.model_path.read_text()
        self.config = config

        # This is the dictionary of data to be passed to Stan
        # Its different to the global Data obejct because this is transformed into numerical values
        # in arrays, as opposed to dataframes or other structures
        self._raw_data: Data | None = None
        self.data = {}
        logger.info(f"Loaded Stan model from {model_path}.")

    def initialise(self, data: Data) -> None:
        self._raw_data = data
        snia = data.filtered_supernova
        self.data = {
            "n_sne": data.num_supernovae,
            "nz_add": data.redshift_simps["nzadd"],
            "n_samples": snia["survey"].n_unique(),
            "redshift_coeffs": data.redshift_coeffs,
            "n_calib": len(data.systematics[0].shape[1]),
            "d_mBx1c_d_calib": data.systematics,
            "n_x1c_star": len(data.redshift_coeffs[0]),
            "threeD_unexplained": int(self.config.threeD_unexplained),
            "mass": snia["mass"].to_numpy(),
            "mass_err": snia["mass_err"].to_numpy(),
            "p_high_mass": snia["p_high_mass"].to_numpy(),
            "in_cluster": snia["in_cluster"].to_numpy(),
            "do_host_mass": int(self.config.do_host_mass),
            "fix_Om": int(self.config.fix_omega_m),
            "MB_by_sample": int(self.config.MB_by_sample),
            "sample_list": snia["sample_index"].to_numpy() + 1,  # Stan uses 1-indexing
            "has_distmod": snia["has_distmod"].to_numpy().astype(int),
            "distmod": snia["distmod"].to_numpy(),
            "zhelio": snia["z_heliocentric"].to_numpy(),
            "redshifts": snia["z_cmb"].to_numpy(),
            "redshifts_sort_fill": data.redshift_simps["redshifts_sort_fill"],
            "unsort_inds": data.redshift_simps["unsort_inds"],
            "obs_mBx1c": data.obs_mBx1c,
            "obs_mBx1c_cov": data.obs_mBx1c_cov,
            "do_blind": int(self.config.do_blinding),
            "do_twoalphabeta": int(self.config.do_two_alpha_beta),
            "outl_frac_prior_lnmean": np.log(self.config.outlier_fraction),
            "outl_frac_prior_lnwidth": 0.5,
            "n_photoz": snia.filter(pl.col("photoz_mean").is_not_null()).height,
            "d_mBx1c_dz_list": data.photoz_uncertainty_dz,
            "photo_z0": snia["photo_z0"].to_numpy(),
            "photo_sigz": snia["photo_sigz"].to_numpy(),
            "photo_spikez": snia["photo_spikez"].to_numpy(),
            "spike_redshift_prob": snia["photo_pspike"].to_numpy(),
            "photoz_inds": snia["photoz_index"].to_numpy(),
            "est_mobs_cuts": snia["est_mobs_cuts"].to_numpy(),
            "est_mobs_sigmas": snia["est_mobs_sigmas"].to_numpy(),
            "mobs_cut0": snia["mobs_cut0"].to_numpy(),
            "mobs_cut1": snia["mobs_cut1"].to_numpy(),
            "BAOCMB_Om_w0_wa_mean": data.bao_cmb_omw0wa["mean"],
            "BAOCMB_Om_w0_wa_covmatrix": data.bao_cmb_omw0wa["cov"],
        }
        logger.info("Stan model initialised with data for fitting.")

    def get_initial_position(self) -> dict[str, int | float | np.ndarray]:
        assert self.data and self._raw_data, "Model data not initialised. Call initialise() first."
        raw, data, config = self._raw_data, self.data, self.config
        n_sne, n_samples = data["n_sne"], data["n_samples"]
        snia = raw.filtered_supernova

        rng = np.random.default_rng()

        return {
            "MB": rng.random(size=n_samples if config.MB_by_sample else 1) * 0.2 - 19.1,
            "MB_slow": rng.random(size=n_samples if config.MB_by_sample else 1) * 0.2 - 19.1,
            "MB_fast_minus_slow": rng.random() * 0.1,
            "Om": 0.3,
            "H0": rng.random() * 5 + 70.0,
            "wDE": -1.01,
            "mu_zbins": rng.normal(size=len(raw.redshift_bins["zbins"])) * 0.05,
            "alpha_angle": np.arctan(rng.random() * 0.2),
            "alpha_angle_fast": np.arctan(rng.random() * 0.2),
            "alpha_angle_slow": np.arctan(rng.random() * 0.2),
            "beta_angle_blue": np.arctan(rng.random() * 0.5 + 2.5),
            "beta_angle_blue_fast": np.arctan(rng.random() * 0.5 + 2.5),
            "beta_angle_blue_slow": np.arctan(rng.random() * 0.5 + 2.5),
            "beta_angle_red_low": np.arctan(rng.random() * 0.5 + 2.5),
            "beta_angle_red_high": np.arctan(rng.random() * 0.5 + 2.5),
            "beta_angle_red_fast": np.arctan(rng.random() * 0.5 + 2.5),
            "beta_angle_red_slow": np.arctan(rng.random() * 0.5 + 2.5),
            "mBx1c_int_variance": np.array([0.9, 0.05, 0.05]),
            "delta_0": rng.random() * 0.05,
            "delta_h": 0.5,
            "step_mass": 10.0,
            "step_width": 0.1,
            "calibs": rng.normal(size=len(raw.systematics[0].shape[1])) * 0.01,
            "true_cB": rng.random(size=n_sne) * 0.02 - 0.01 + np.clip(snia["color"].to_numpy() / 2.0, -0.2, 1.0),
            "true_cR_unit": rng.random(size=n_sne) * 0.5 + 0.5,
            "true_x1": rng.random(size=n_sne) * 0.2 - 0.1 + snia["x1"],
            "x1_star": rng.random(size=data["n_x1c_star"]) * 0.5,
            "tau_x1": -rng.random(size=data["n_x1c_star"]),
            "R_x1": rng.random(size=data["n_x1c_star"]) * 0.5 + 0.25,
            "x1_star_fast": rng.random() * 0.5 - 1.25,
            "x1_star_slow": rng.random() * 0.5,
            "R_x1_fast": rng.random() * 0.25 + 0.4,
            "R_x1_slow": rng.random() * 0.25 + 0.4,
            "c_star": -rng.random(size=data["n_x1c_star"]) * 0.05,
            "c_star_fast": -rng.random() * 0.05,
            "c_star_slow": -rng.random() * 0.05,
            "tau_c": rng.random(size=data["n_x1c_star"]) * 0.05 + 0.02,
            "R_c": rng.random(size=data["n_x1c_star"]) * 0.05 + 0.02,
            "outl_frac": rng.random() * 0.02 + 0.01,
            "mobs_cuts": data["est_mobs_cuts"] + rng.normal(size=n_samples) * 0.1,
            "mobs_cut_sigmas": [0.5] * n_samples,
            "dz": rng.normal(size=data["n_photoz"]) * 0.01,
        }
