from pathlib import Path
import polars as pl
import numpy as np
from unity import Config, Data, logger, CosmologyModel
from astropy.cosmology import FlatLambdaCDM, w0waCDM

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
        elif model_path.suffix == ".py":
            return NumpyroModel(model_path, config)
        else:
            raise ValueError(f"Unsupported model file extension: {model_path.suffix}")


class StanModel(Model):
    def __init__(self, model_path: Path, config: Config):
        assert model_path.exists(), f"Model file {model_path} does not exist."
        self.model_path = model_path
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

        blinding_mapping = {
                'none':0,
                'fiducial':1,
                'stochastic':2
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
            "blinding":blinding_mapping[self.config.blinding],
            "really_unblind":int(self.config.really_unblind),
            #"do_blind": int(self.config.do_blinding),
            #"blinding": str(self.config.blinding),
            #"really_unblind": str(self.config.really_unblind),
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

    def blind(self, kind: str = 'fiducial', MB_blind: float = -19.1, alpha_blind: float = 0.14, beta_blind: float = 3.1) -> None:
        '''
        Perform calibrator distance and per-sample blinding.

        kind == 'fiducial':   blind against a fixed, known cosmology read from a precomputed
                              (z, mu) grid file. Reproducible / unblind-able, so runs on the
                              same data are comparable.
        kind == 'stochastic': blind against a freshly randomized cosmology generated at runtime
                              and NEVER saved, so the resulting chains cannot be unblinded.
                              For debugging / SN-modeling checks only.

        TODO: User-defined path to a fiducial cosmology file. Currently hard-coded path points to David's .txt files.
        '''

        from scipy.interpolate import CubicSpline

        if kind == 'fiducial':
            # Load the fine-grid (z, mu) fiducial cosmology and interpolate it (a la original UNITY)
            if self.config.cosmology_model is CosmologyModel.OM_W0_WA:
                zblind, mublind, NA = np.genfromtxt(f'{self.config.data_dir}/blinding_cosmologies/z_mu_dmudOm_w0wa.txt', unpack=True)
            elif self.config.cosmology_model is CosmologyModel.OM:
                zblind, mublind, NA = np.genfromtxt(f'{self.config.data_dir}/blinding_cosmologies/z_mu_dmudOm.txt', unpack=True)
            else:
                raise ValueError(f"Fiducial blinding not supported for cosmology model {self.config.cosmology_model}.")
            mu_blinding_fiducial = CubicSpline(zblind, mublind)

        elif kind == 'stochastic':
            # Draw a random cosmology at runtime; never saved, so chains cannot be unblinded.
            H0_stoch = np.random.uniform(low=60, high=80)
            Om_stoch = np.random.uniform(low=0.25, high=0.35)
            if self.config.cosmology_model is CosmologyModel.OM_W0_WA:
                w0_stoch = np.random.uniform(low=-1.5, high=-0.5)
                wa_stoch = np.random.uniform(low=-3, high=1)
                cosmo_fiducial = w0waCDM(H0=H0_stoch, w0=w0_stoch, wa=wa_stoch, Om=Om_stoch)
            elif self.config.cosmology_model is CosmologyModel.OM:
                cosmo_fiducial = FlatLambdaCDM(H0=H0_stoch, Om0=Om_stoch)
            else:
                raise ValueError(f"Stochastic blinding not supported for cosmology model {self.config.cosmology_model}.")

            def mu_blinding_fiducial(z, cosmo_fiducial=cosmo_fiducial):
                return 5 * np.log10(cosmo_fiducial.luminosity_distance(z).to('pc').value / 10)

        else:
            raise ValueError(f"Unknown blinding kind '{kind}'. Expected 'fiducial' or 'stochastic'.")

        # Now carry out blinding on calibrator distances. Only the calibrators (has_distmod == 1)
        # carry a real distmod; the rest are 0 and ignored by Stan, so we shift only those.
        target_distmod = mu_blinding_fiducial(self.data["redshifts"])
        calib = self.data["has_distmod"] == 1
        med_offset = np.median(target_distmod[calib] - self.data["distmod"][calib])
        self.data["distmod"][calib] += med_offset

        # And now the per-sample hubble flow blinding. TJH: Almost faithful replicate of DR code; I just removed the redundant computation of target_distmod in the H_resid line
        for iter_count in range(2):
            # Compute real observed moduli (according to whatever mB normalization Union's LC fitting is set to)
            #print(self.data['obs_mBx1c'])
            mB, x1, c = np.array(self.data['obs_mBx1c']).T
            muobs = mB + alpha_blind*x1 - beta_blind*c - MB_blind
            #muobs = self.data["obs_mBx1c"][:,0] + alpha_blind*self.data["obs_mBx1c"][:,1] - beta_blind*self.data["obs_mBx1c"][:,2] - MB_blind

            # Compute the real hubble residuals relative to the fiducial cosmology
            H_resid = muobs - target_distmod #mublindfn(self.data["redshifts"])

            # Compute approximate diagonal uncertainties TJH to DR: Is this used anywhere?
            #dmuobs = np.sqrt(0.15**2. + self.data["obs_mBx1c_cov"][:,0,0] + alpha_blind**2. * self.data["obs_mBx1c_cov"][:, 1,1] + beta_blind**2. * self.data["obs_mBx1c_cov"][:, 2,2]) # Doesn't have to be exact

            for sample_ind in range(self.data["n_samples"]):
                sel_hubble_flow = self.data['has_distmod']==0
                inds = np.where((self.data["sample_list"] == sample_ind)*sel_hubble_flow)

                if len(inds[0]) > 0:
                    med_HR = np.median(H_resid[inds])

                    inds = np.where((self.data["sample_list"] == sample_ind))

                    for SN_ind in inds[0]:
                        # TJH: Sam has mBx1c stored as a list of len=3 arrays, so can't do slicing. We are already looping so it's easy.
                        self.data['obs_mBx1c'][SN_ind][0] -= med_HR

                        # TJH: mB_list has been removed as key from the data container. Presumably because of redundancy. 
                        # Double-check to make sure I don't let unblinded stuff slip through.
                        #self.data["mB_list"][SN_ind] -= med_HR

                    if iter_count > 0:
                        assert abs(med_HR) < 1e-3




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
        from cmdstanpy import CmdStanModel

        logger.info(f"Loading Stan model from {self.model_path} for MCMC sampling.")
        stan_model = CmdStanModel(stan_file=self.model_path)
        inits = [self.get_initial_position() for _ in range(self.config.num_chains)]
        logger.info(f"Starting {self.config.num_chains} Stan samplers, each for {self.config.iterations} iterations...")
        fit = stan_model.sample(
            data=self.data,
            chains=self.config.num_chains,
            parallel_chains=self.config.num_chains,
            inits=inits,  # type: ignore
            iter_warmup=self.config.warmup_iterations,
            iter_sampling=self.config.iterations,
            refresh=self.config.refresh_iterations,
        )

        logger.info("Stan MCMC sampling complete. Extracting samples.")
        if self.config.extra_single_dimension_parameters_only:
            # draws_pd(vars=...) accepts method vars (lp__, ...) and *base* Stan variable
            # names, but NOT indexed columns like "MB_slow[1]". Keeping only bracket-free
            # names leaves scalar params + diagnostics, which are all valid vars.
            columns = [c for c in fit.column_names if "[" not in c]
            logger.info(
                f"Saving only single-dimension parameters: {len(columns)} out of {len(fit.column_names)} total parameters."
            )
            df = fit.draws_pd(vars=columns)
        else:
            # Save everything. Passing the full column_names to vars= raises
            # "Unknown variable: <name>[i]" on indexed columns, so call draws_pd() with
            # no vars, which returns all draws (chain__/iter__/draw__ + every column).
            df = fit.draws_pd()

        logger.info(
            f"Completed MCMC fitting with {self.config.num_chains} chains, "
            f"warmup {self.config.warmup_iterations}, and {self.config.iterations} iterations."
        )
        return pl.from_pandas(df)


class NumpyroModel(StanModel):
    """Samples a NumPyro port of the model (e.g. unity_1_8_numpyro.py) with NUTS.

    Reuses StanModel's data preparation and blinding wholesale: initialise()
    builds the identical Stan data dict, which the model file's make_model(data)
    consumes (the posterior was validated against Stan end-to-end on 2026-07-10:
    log-density parity ~9e-15 via BridgeStan, full-sample sampling comparison
    statistically indistinguishable).

    The returned DataFrame matches StanModel.fit()'s mcmc_samples.parquet layout
    (diagnostic columns, Stan CSV parameter naming, chains concatenated in order,
    no chain__ column) with two caveats:
    - lp__ is -potential_energy: it differs from Stan's lp__ by the two engines'
      different simplex-transform Jacobians (parameter posteriors are invariant;
      never cross-compare lp__ between engines).
    - per-SN transformed parameters (model_mu, true_cR, outl/inl_loglike_by_SN,
      ...) and last-SN loop-leftover scalars (beta_eff, p_high_mass*,
      this_delta_h, this_norm_LL_*, dz_*) are not computed, so downstream
      PPD/plot scripts that need them require the Stan model. The cheap
      transformed scalars (alpha_*, beta_*, this_MB_slow) ARE emitted for
      layout compatibility.

    Tuning note: NumPyro's unconstraining transforms differ from Stan's, so its
    warmup requirement is its own number — warmup 1250 sufficed for Stan on the
    full 2085-SN config but trapped a NumPyro chain; use warmup_iterations >= 2500.
    """

    def __init__(self, model_path: Path, config: Config):
        assert model_path.exists(), f"Model file {model_path} does not exist."
        self.model_path = model_path
        self.config = config
        self._raw_data: Data | None = None
        self.data = {}
        logger.info(f"Loaded NumPyro model from {model_path}.")

    def fit(self) -> pl.DataFrame:
        import importlib
        import os

        config = self.config
        # Backend selection must happen before JAX initializes, hence the
        # env var (not jax.config) and the function-local imports.
        if config.jax_device is not None:
            os.environ.setdefault("JAX_PLATFORMS", config.jax_device)
        import numpyro

        if config.chain_method == "parallel":
            # gives the CPU backend one XLA device per chain; no-op on GPU
            numpyro.set_host_device_count(config.num_chains)
        import jax

        jax.config.update("jax_enable_x64", True)  # model is float64-only
        from numpyro.infer import MCMC, NUTS

        if config.warmup_iterations < 2000:
            logger.warning(
                f"warmup_iterations={config.warmup_iterations} is risky for NumPyro on UNITY "
                "models: 1250 left a chain trapped below the main mode on the full config. "
                "Use >= 2500 and check per-chain lp__ agreement."
            )

        module = importlib.import_module(f"unity.models.{self.model_path.stem}")
        model = module.make_model(self.data)
        seed = config.sampling_seed
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")
        logger.info(
            f"Starting NumPyro NUTS: {config.num_chains} chains ({config.chain_method}), "
            f"warmup {config.warmup_iterations}, {config.iterations} iterations, "
            f"seed {seed}, devices {jax.devices()}."
        )
        mcmc = MCMC(
            NUTS(model),
            num_warmup=config.warmup_iterations,
            num_samples=config.iterations,
            num_chains=config.num_chains,
            chain_method=config.chain_method,
            progress_bar=config.refresh_iterations > 0,
        )
        mcmc.run(
            jax.random.PRNGKey(seed),
            extra_fields=("potential_energy", "energy", "num_steps",
                          "accept_prob", "diverging", "adapt_state.step_size"),
        )
        # pmap dispatch is async; block before touching wall clocks or results
        samples = jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
        extra = mcmc.get_extra_fields(group_by_chain=True)
        logger.info("NumPyro MCMC sampling complete. Extracting samples.")

        flat = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        n_leapfrog = flat(extra["num_steps"])
        cols: dict[str, np.ndarray] = {
            "lp__": -flat(extra["potential_energy"]),
            "accept_stat__": flat(extra["accept_prob"]),
            "stepsize__": flat(extra["adapt_state.step_size"]),
            "treedepth__": np.ceil(np.log2(n_leapfrog + 1)),
            "n_leapfrog__": n_leapfrog,
            "divergent__": flat(extra["diverging"]),
            "energy__": flat(extra["energy"]),
        }
        divergences = int(cols["divergent__"].sum())
        lp_chain = np.asarray(extra["potential_energy"]).mean(axis=1) * -1.0
        logger.info(f"Divergences: {divergences}. Per-chain lp__ means: "
                    + " ".join(f"{m:.1f}" for m in lp_chain))
        if np.ptp(lp_chain) > 2.0 * np.asarray(extra["potential_energy"]).std(axis=1).mean():
            logger.warning("Per-chain lp__ means are far apart relative to the within-chain "
                           "spread — a chain is likely trapped (increase warmup_iterations).")

        # Stan CSV naming: vector sites get name[i] (1-based) even for length 1,
        # scalar sites are bracket-free — so the extra_single_dimension filter
        # below behaves exactly like the StanModel one.
        params: dict[str, np.ndarray] = {}
        for name, _, shape in module.param_spec(self.data):
            if name not in samples:
                continue
            x = np.asarray(samples[name], dtype=np.float64)
            x = x.reshape(x.shape[0] * x.shape[1], -1)
            if shape == ():
                params[name] = x[:, 0]
            else:
                for j in range(x.shape[1]):
                    params[f"{name}[{j + 1}]"] = x[:, j]

        # transformed scalars the Stan runs save (cheap, derived from raw draws)
        derived: dict[str, np.ndarray] = {
            "alpha_fast": np.tan(params["alpha_angle_fast"]),
            "alpha_slow": np.tan(params["alpha_angle_slow"]),
        }
        beta_blue = "beta_angle_blue" if self.data["do_twoalphabeta"] else "beta_angle_red_low"
        beta_high = "beta_angle_red_high" if self.data["do_twoalphabeta"] else "beta_angle_red_low"
        derived["beta_B"] = np.tan(params[beta_blue])
        derived["beta_R_low"] = np.tan(params["beta_angle_red_low"])
        derived["beta_R_high"] = np.tan(params[beta_high])
        if not self.data["MB_by_sample"]:
            derived["this_MB_slow"] = params["MB_slow[1]"]

        if config.extra_single_dimension_parameters_only:
            params = {k: v for k, v in params.items() if "[" not in k}
        cols |= params | derived

        logger.info(
            f"Completed NumPyro MCMC fitting with {config.num_chains} chains, "
            f"warmup {config.warmup_iterations}, and {config.iterations} iterations."
        )
        return pl.DataFrame(cols)
