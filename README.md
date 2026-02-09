# UNITY

A package for performing supernova cosmology using Bayesian Hierarchical Models

## Installation


For convenience, many of the top-level administrative functions are collected into a Makefile. To install this repository, simply run `make install`. 

Please note that Stan 3 does not make it super easy to install on anything other than Linux. I've enabled the no-binary flag 
on the pystan dependency in the pyproject.toml file to try and force a rebuild, but I'm not sure if it will work.

Please see [the Apple specific instructions](https://pystan.readthedocs.io/en/latest/faq.html#how-can-i-run-pystan-on-macos-with-apple-silicon-chips-apple-m1-m2-etc) if it fails.

Alternatively, there is a way to build an image of the repository which may be easier to run. Simply `make image` and hopefully an image will be built and then there's an example of running it in the Makefile, accessible via `make run_image`.


## Running Unity

To kick off a run with the default configuration, try `make run`, which is just a shortcut for `uv run union3`. 

You can customise what runs in a few ways.

1. If you have a config file with overrides in `src/union3/configs`, you can pass in the filename, like `uv run union3 base=union3.0.yml`
2. If you want temporary overrides, you can pass them in, like `uv run union3 --filters.max_redshift 0.3` (see `config.py` for all the options)
3. You can also configure what's run via environment variables, which is especially useful when running via an image.

```bash
export FILTERS__MAX_REDSHIFT=0.3
uv run union3
```

Finally, the default log level is probably `INFO`. If you want to see more detail, you can control loguru's level with the `LOGURU_LEVEL` env var, so you could run `export LOGURU_LEVEL=DEBUG` to see more logs.






*****

# OLD README BELOW


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
