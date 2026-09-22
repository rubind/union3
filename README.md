# UNITY

A package for performing supernova cosmology using Bayesian Hierarchical Models

## Installation


For convenience, many of the top-level administrative functions are collected into a Makefile. To install this repository, simply run `make install`. 

Please note that Stan 3 does not make it super easy to install on anything other than Linux. I've enabled the no-binary flag 
on the pystan dependency in the pyproject.toml file to try and force a rebuild, but I'm not sure if it will work.

Please see [the Apple specific instructions](https://pystan.readthedocs.io/en/latest/faq.html#how-can-i-run-pystan-on-macos-with-apple-silicon-chips-apple-m1-m2-etc) if it fails.

Alternatively, there is a way to build an image of the repository which may be easier to run. Simply `make image` and hopefully an image will be built and then there's an example of running it in the Makefile, accessible via `make run_image`.


## Running Unity

To kick off a run with the default configuration, try `make run`, which is just a shortcut for `uv run unity`.

You can customise what runs in a few ways.

1. If you have a config file with overrides in `src/unity/configs`, you can pass in the filename, like `uv run unity --base union31_H0_snOnly.yml`
2. If you want temporary overrides, you can pass them in, like `uv run unity --filters.max_redshift 0.3` (see `config.py` for all the options)
3. You can also configure what's run via environment variables, which is especially useful when running via an image.

```bash
export FILTERS__MAX_REDSHIFT=0.3
uv run unity
```

Finally, the default log level is probably `INFO`. If you want to see more detail, you can control loguru's level with the `LOGURU_LEVEL` env var, so you could run `export LOGURU_LEVEL=DEBUG` to see more logs.

## Repository layout

Not everything here is part of the current pipeline. In short:

| Path | Status |
| --- | --- |
| `src/unity/` | The pipeline. Config, data loading, models, sampling. |
| `src/unity/configs/` | Committed run configs, selected with `--base`. |
| `src/legacy/lcfit_extraction/` | **Still live**, despite the directory name: turns raw per-survey light-curve fits into the parquet files the loader reads. |
| `scripts/numpyro_port/` | **Still live**: the Stan-vs-NumPyro parity harness, imported at runtime by `src/unity/models/jax_unity.py`. |
| `scripts/` | Mostly UNITY 1.5-era and kept for reference, plus a few standalone tools for the current pipeline such as `check_mu_matrix.py`. Nothing here is driven by `uv run unity`. |
| `other_cosmology/` | UNITY 1.5-era, kept for reference and for reproducing published products. |

## Data release

A binned distance-modulus release is produced directly by a run, by setting `write_mu_matrix: true`
together with `cosmology_model: "binned_mu"` and `extra_vector_parameters_to_save: ["mu_zbins"]`.
The committed config `union31_unity18_published_binnedMu.yml` does this for the published
Union3.1 / UNITY 1.8 result, so it can be regenerated with:

```bash
uv run unity --base union31_unity18_published_binnedMu.yml
```

Two files land in the run's `output_dir` alongside the chains:

- `mu_mat.fits` — one `(n_bins + 1) x (n_bins + 1)` float64 image in the packed layout used by past
  Union releases. Row and column zero are metadata, not part of the matrix: `[0, 1:]` holds the bin
  redshifts, `[1:, 0]` the binned distance moduli, and `[1:, 1:]` the inverse covariance. The header
  records which run produced it, including the sampling seed.
- `mu_binned.ecsv` — the same content as a table, with the covariance rather than its inverse.

### Conventions

The distance moduli are residuals relative to `FlatLambdaCDM(H0=70, Om0=0.3)`. Add that fiducial
back to recover absolute distance moduli. That step is exact at the bin redshifts, since each
interpolation basis function is 1 at its own node and 0 at the others.

The same fiducial also sets the shape within a bin: a supernova between two nodes is modelled as
the fiducial plus the interpolated residual, and adding the fiducial back does not remove that. The
effect stays under the statistical error. Swapping the fiducial for `Om0=0.35` leaves a difference
the basis cannot absorb of 0.004 mag rms below z = 0.8, against a per-bin sigma of 0.029 mag, and
0.12-0.15 sigma in the bins at z = 0.996-1.232, 0.37 sigma at z = 1.391, and 0.60 sigma in the top
bin at z = 2.2623, which holds two supernovae and carries sigma = 0.34 mag. The bins where the
fiducial shape matters most are the ones whose statistical errors are largest. Doubling the swap to
`Om0=0.4` roughly doubles each ratio.

Cosmology fits against these bins should carry a free magnitude offset (scriptM). The bins are free
parameters with no cosmological relation imposed between them, and `Om`, `wDE` and `waDE` are not
fit in a `binned_mu` run, but their overall normalisation is not free. Each interpolation basis function in
`_get_redshift_bins` takes the value -1 at z = 0, so below the first bin (z = 0.05) a uniform shift
of every bin is not a rigid shift of the model: the response reaches about -15 at z = 0.01, where
574 of the 2085 supernovae sit. That anchor holds the mean of the bins near zero (-0.0048 +/- 0.0042
mag in the published run) and is why the overall-offset mode of the covariance is narrow. It is a
normalisation, not a measurement of absolute scale, which would require calibrator distances or a
sound-horizon prior.

This differs from UNITY 1.5, which anchored at 0 rather than -1 (`scripts/read_and_sample.py:668`).
There the offset direction was nearly degenerate with `MB` and ran away, leaving the 1.5-era
matrices about 0.093 mag wide in it against 0.0042 mag here. The shapes are unaffected: dropping
the common mode moves the median per-bin sigma by 0.0001 mag, and the published Union3.1 / UNITY 1.8
bins differ from the 1.5 product by a pure constant.

Runs are blinded by default. The flags that disable blinding are deliberately never written into
any committed config; they are typed on the command line for each run, so that no file in this
repository is a paste-able unblinding template.






---

# Retired: the UNITY 1.5 pystan flow

Everything below this line describes the **retired** UNITY 1.5 analysis: a pystan 2.19 pipeline
driven by `read_and_sample.py` and paramfiles, superseded by the `uv run unity` flow documented
above. None of it applies to the installed package, and the scripts it names live under `scripts/`
and `other_cosmology/` rather than in `src/unity/`.

It is kept because it is the only written record of several one-off regeneration procedures that
are still occasionally needed: the bulk-flow eigenvectors, the CMB compression, and the fiducial
sound horizon used when updating BAO. Read it as history, not as instructions.

# union3
Union3/UNITY1.5 repo

# UNITY1.5 requires:

```numpy==1.22.4```
```pystan==2.19.1.1```
```cython==3.0.10```

# Need to set these environment variables:

```export UNITY=path/to/union3```

```export UNION=path/to/lcfits```

```export PATHMODEL=path/to/salt```

# For making the LC fits:

From Union3_Photometry:

```python parse_nearby.py paramfile_default.txt```

If a host galaxy or redshift is missing from the json, use:

```python make_list_of_hosts.py LSQ14fep 0.06 0.01 PESSTO```

or

```python make_list_of_hosts.py LSQ13aiz "ESO 576-17"```

From the Union3 directory, run:

```python $PATHMODEL/python_code/tmpfit.py 0 4 1```

where 4 is the number of LC fits to do at once.

```python $PATHMODEL/python_code/cutfits.py [optional directories to look in]```

For making the v1 files of SNe that pass cuts. Note that cutfits.py takes arguments if you only want to check some directories but not others.

# For running UNITY:

```python read_and_sample.py paramfile_Union3.txt 1```

'1' is the cosmology model (flat LCDM, also available are flat Om-w, flat-w0wa with a BAO+CMB prior, and spline mu(z) for releasing distances). read_and_sample.py also makes a pickle file that can be used as input:

```python read_and_sample.py inputs_XXX.pickle 1```

This is much more portable/reproducable than making read_and_sample.py read in LC fits.

# For updating filters or magnitude systems (magsys):

Union3_Photometry/NB99_also_contains_non_X-CALIBUR_magsys/AB_Landolt.py contains the color-transformed magnitude systems.

For magnitude systems and instruments: edit the original_* files in $PATHMODEL. Then run python_code/shiftfilters.py to make the files that are actually read in by SALT.

# For regenerating the bulk-flow eigenvectors:

From BulkFlow/Public/Run_Example2:

```python step1_make_Union3.py path/to/union3/```

```./pairV table.input table.input2```

```python step2_convert_to_fits.py```

```python step3_eig_reduce_n_at_a_time.py 1 100```

# For updating the CMB compression:

This merges the chains into one file, and unpacks the duplicated samples:

```python step1_make_new_samps.py```

```python step2_make_cov_mat.py```

# For getting fidicial r_d for updating BAO:

```python step2_standard_BAO.py```

# For evaluating phase cuts:

```set_up_LC_tests.py```

```slurm_phase_testing.py```

# For doing simulated-data testing:

```python $UNITY/simulated/step1_make_simLCs.py --ndataset 1 --addnoise 1 --addcalibration 1 --modeluncertainty 1 --prefixname sim_H0 --skewdist 1 --volumelimited 0 --obsmagselection 1 --zrangekeys LHV```
```python $UNITY/simulated/step1_make_simLCs.py --ndataset 1 --addnoise 1 --addcalibration 1 --modeluncertainty 1 --prefixname sim_H0 --skewdist 1 --volumelimited 0 --obsmagselection 1 --zrangekeys SLHV --nnearbyperset 300 --ncalibperset 21 --sigzp 0.01```

```python $UNITY/simulated/step1B_Om_sampled_well.py```
