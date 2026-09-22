from unity import logger, Config, Data, Model
import polars as pl

from unity.plotting import plot_hubble_diagram_from_stanInputData, plot_cosmology_constraints


def _release_provenance(config: Config, model: Model, samples: pl.DataFrame) -> dict[str, object]:
    """Which run produced a data-release product, recorded into the product itself."""
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal to a finished run
        commit = "unknown"

    return {
        "config": config.base or "(defaults)",
        "git_commit": commit,
        "fit_model": config.fit_model,
        "sampler": str(config.sampler),
        "cosmology_model": str(config.cosmology_model),
        "sampling_seed": getattr(model, "sampling_seed_used", None) or config.sampling_seed or "(fresh random)",
        "num_chains": config.num_chains,
        "warmup_iterations": config.warmup_iterations,
        "iterations": config.iterations,
        "n_draws_total": samples.height,
        "n_sne": int(model.data["n_sne"]),
        "ordered_beta": config.ordered_beta,
        "blinding": config.blinding,
        "distance_ladder_file": str(config.distance_ladder_file),
    }


def fit_cosmology(config: Config | None = None) -> pl.DataFrame | None:
    if config is None:
        config = Config()
    logger.info(f"Running Unity with base config file: {config.base}")
    logger.info(f"Run settings: {config.model_dump_json(indent=2)}")

    data = Data.from_config(config)
    print(type(data))
    #print('DATA: No. of SNe that have an external distance:', data.all_supernova['has_distmod'].sum())
    model = Model.from_config(config)
    model.initialise(data)

    # Blinding block 
    # TODO: Actual interruption prompt asking for confirmation from user. Sam will know a good way.
    # TODO: Check that blinding was successful. David has assertion blocks for this.
    if config.blinding != 'none':
        print(f'Blinding the cosmology according to {config.blinding} protocol.')
        model.blind(kind=config.blinding)
    else:
        print('INITIATING FULLY UNBLINDED RUN. ARE YOU SURE? Checking the double-check parameter now...')
        if config.really_unblind:
            print('Unblinding confirmed. Proceeding with UNBLINDED sampling.')
        else:
            print('Unblinding rejected. If you are sure you want to unblind, set "really_unblind" to True upon runtime.')
            print('Now blinding...')
            model.blind(kind='fiducial')

    # Moved this block before sampling, because it's just a sanity check on LC fitting (and now on blinding too).
    # TODO: If a blinded run, scramble the signs and redshifts within each survey, so we can't identify individual SNe
    if config.do_plotting:
        print('Plotting approximate Hubble diagram from input LC fit data. Will match blinding protocol set for UNITY.')
        plot_hubble_diagram_from_stanInputData(model, config)


    print('No. of SNe that have an external distance:', model.data['has_distmod'].sum())
    samples = model.fit()

    # TODO: make this path configurable and part of the config
    samples.write_parquet(config.output_dir / "mcmc_samples.parquet")

    if config.write_mu_matrix:
        # Written here rather than from a standalone CLI because zbins is never persisted and
        # depends on the exact filtered SN set, so it cannot be re-derived safely after the fact.
        from unity.mu_matrix import write_release_products

        # Never let a product-writing failure discard a finished run: the chains are already on
        # disk above, and a multi-hour sample is far more expensive than a rebuildable product.
        try:
            write_release_products(
                samples,
                model.data["zbins"],
                config.output_dir,
                provenance=_release_provenance(config, model, samples),
            )
        except Exception:
            logger.exception(
                "Failed to write the binned-mu release products. The chains are safe in "
                f"{config.output_dir / 'mcmc_samples.parquet'}; rebuild the products from them "
                "with unity.mu_matrix.write_release_products once the cause is fixed."
            )

    # describe() materializes per-column stats; on all-latents outputs (100k+ columns)
    # it pegs a core for over an hour at ~35GB RSS, so only summarize narrow outputs.
    if samples.width <= 1000:
        print(samples.describe())
    else:
        print(f"Samples: {samples.height} rows x {samples.width} columns (describe() skipped for wide output)")
    if config.do_plotting:
        plot_cosmology_constraints(config, samples)
    return samples


if __name__ == "__main__":
    from rich.logging import RichHandler

    logger.configure(handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}])
    fit_cosmology()
