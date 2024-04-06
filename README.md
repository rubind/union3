# union3
Union3/UNITY1.5 repo

# UNITY1.5 requires:

```numpy==1.22.4```
```pystan==2.19.1.1```
```cython==3.0.10```

# Need to set these environment variables:

```export UNITY=path/to/union3```

```export UNION=path/to/lcfits```

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

```python $PATHMODEL/python_code/cutfits.py```

For making the v1 files of SNe that pass cuts. Note that cutfits.py takes arguments if you only want to check some directories but not others.

# For updating filters or magnitude systems (magsys):

Union3_Photometry/NB99_also_contains_non_X-CALIBUR_magsys/AB_Landolt.py contains the color-transformed magnitude systems.

For magnitude systems and instruments: edit the original_* files in $PATHMODEL. Then run python_code/shiftfilters.py to make the files that are actually read in by SALT.

# For regenerating the bulk-flow eigenvectors:

From BulkFlow/Public/Run_Example2:

```python step1_make_Union3.py```

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

```simulated/step1_make_simLCs.py 1 1```
