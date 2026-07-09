from unity import logger, Config, Data, Model
import polars as pl

from unity.plotting import plot_hubble_diagram_from_stanInputData, plot_cosmology_constraints


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

    print(samples.describe())
    if config.do_plotting:
        plot_cosmology_constraints(config, samples)
    return samples


if __name__ == "__main__":
    from rich.logging import RichHandler

    logger.configure(handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}])
    fit_cosmology()
