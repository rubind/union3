from pathlib import Path
import polars as pl
import numpy as np
from union3 import Config, Data, logger, CosmologyModel


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
        cosmo_model_mapping = {
            CosmologyModel.OM: 1,
            CosmologyModel.BINNED_MU: 2,
            CosmologyModel.OM_W: 3,
            CosmologyModel.Q0_J0: 4,
            CosmologyModel.OM_W0_WA: 5,
            CosmologyModel.BINNED_MU_COMOVING_INTERPOLATION: 6,
        }
        self.data = {
            "cosmo_model": cosmo_model_mapping[self.config.cosmology_model],
            "n_sne": data.num_supernovae,
            "nzadd": data.redshift_simps["nzadd"],
            "n_samples": data.samples.height,
            "redshift_coeffs": data.redshift_coeffs,
            "n_calib": data.systematics[0].shape[1],
            "d_mBx1c_d_calib": data.systematics,
            "n_x1c_star": len(data.redshift_coeffs[0]),
            "threeD_unexplained": int(self.config.threeD_unexplained),
            "mass": snia["mass"].to_numpy().astype(np.float64),
            "mass_err": snia["mass_err"].to_numpy().astype(np.float64),
            "p_high_mass": snia["p_high_mass"].to_numpy().astype(np.float64),
            "in_cluster": snia["in_cluster"].to_numpy(),
            "do_host_mass": int(self.config.do_host_mass),
            "fix_Om": int(self.config.fix_omega_m),
            "MB_by_sample": int(self.config.MB_by_sample),
            "sample_list": snia["sample_index"].to_numpy().astype(int) + 1,  # Stan uses 1-indexing
            "has_distmod": snia["has_distmod"].to_numpy().astype(int),
            "distmod": snia["distmod"].to_numpy().astype(np.float64),
            "zhelio": snia["z_heliocentric"].to_numpy().astype(np.float64),
            "redshifts": snia["z_cmb"].to_numpy().astype(np.float64),
            "redshifts_sort_fill": data.redshift_simps["redshifts_sort_fill"],
            "unsort_inds": data.redshift_simps["unsort_inds"],
            "obs_mBx1c": data.obs_mBx1c,
            "obs_mBx1c_cov": data.obs_mBx1c_cov,
            "do_blind": int(self.config.do_blinding),
            "do_twoalphabeta": int(self.config.do_two_alpha_beta),
            "outl_frac_prior_lnmean": float(np.log(self.config.outlier_fraction)),
            "outl_frac_prior_lnwidth": 0.5,
            "n_photoz": int(snia.filter(pl.col("photo_z0").is_not_null()).height),
            "d_mBx1c_dz_list": data.photoz_uncertainty_dz,
            "photo_z0": snia["photo_z0"].drop_nulls().to_numpy(),
            "photo_sigz": snia["photo_sigz"].drop_nulls().to_numpy(),
            "photo_spikez": snia["photo_spikez"].drop_nulls().to_numpy(),
            "spike_redshift_prob": snia["photo_pspike"].drop_nulls().to_numpy(),
            "photoz_inds": snia["photoz_index"].to_numpy().astype(np.int32),
            "est_mobs_cuts": data.samples["est_mobs_cuts"].to_numpy(),
            "est_mobs_sigmas": data.samples["est_mobs_sigmas"].to_numpy(),
            "mobs_cut0": snia["mobs_cut0"].to_numpy(),
            "mobs_cut1": snia["mobs_cut1"].to_numpy(),
            "BAOCMB_Om_w0_wa_mean": data.bao_cmb_omw0wa["mean"],
            "BAOCMB_Om_w0_wa_covmatrix": data.bao_cmb_omw0wa["cov"],
            "n_zbins": len(data.redshift_bins["zbins"]),
            "zbins": data.redshift_bins["zbins"],
            "dmu_dbin": data.redshift_bins["dmu_dbin"],
            "dmudz_dbin": data.redshift_bins["dmudz_dbin"],
            "mu_const": snia["mu_const"].to_numpy(),
            "dmu_const_dz": snia["dmu_const_dz"].to_numpy(),
        }
        logger.info("Stan model initialised with data for fitting.")

    def get_initial_position(self) -> dict[str, int | float | np.ndarray]:
        assert self.data and self._raw_data, "Model data not initialised. Call initialise() first."
        raw, data, config = self._raw_data, self.data, self.config
        n_sne, n_samples = data["n_sne"], data["n_samples"]
        snia = raw.filtered_supernova

        rng = np.random.default_rng()

        position = {
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
            "calibs": rng.normal(size=data["n_calib"]) * 0.01,
            "true_cB": rng.random(size=n_sne) * 0.02 - 0.01 + np.clip(snia["color"].to_numpy() / 2.0, -0.2, 1.0),
            "true_cR_unit": rng.random(size=n_sne) * 0.5 + 0.5,
            "true_x1": rng.random(size=n_sne) * 0.2 - 0.1 + snia["x1"].to_numpy(),
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
        return position

    def fit(self) -> pl.DataFrame:
        import stan

        stan_model = stan.build(self.model_text, data=self.data)
        init = [self.get_initial_position() for _ in range(self.config.num_chains)]
        logger.info(f"Starting {self.config.num_chains} Stan samplers, each for {self.config.iterations} iterations...")
        fit = stan_model.sample(
            num_chains=self.config.num_chains,
            init=init,
            num_samples=self.config.iterations,
            num_warmup=self.config.warmup_iterations,
        )

        logger.info("Stan MCMC sampling complete. Extracting samples.")
        params = [param for param, dim in zip(fit.param_names, fit.dims) if not dim or dim == [1]]
        if not self.config.extra_single_dimension_parameters_only:
            df_full = fit.to_frame()
            df = pl.from_pandas(df_full[df_full.columns[: self.config.max_params_to_save]])
        else:
            df = pl.DataFrame({p: fit[p].flatten() for p in params})

        logger.info(
            f"Completed MCMC fitting with {self.config.num_chains} chains, "
            f"warmup {self.config.warmup_iterations}, and {self.config.iterations} iterations."
        )
        return df
