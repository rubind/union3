# union3
Union3/UNITY1.5 repo

```export UNITY=path/to/union3```

```export UNION=path/to/lcfits```

For the LC fits:

```python parse_nearby.py paramfile_default.txt```

Note that NB99_also_contains_non_X-CALIBUR_magsys/AB_Landolt.py contains the color-transformed magnitude systems.

For magnitude systems and instruments: edit the original_* files in $PATHMODEL. Then run python_code/shiftfilters.py if you want to be able to use snfit (as opposed to SALT3.py).

From the Union3 directory, run:

```python $PATHMODEL/python_code/tmpfit.py 0 4 1```

For regenerating the bulk-flow eigenvectors: from BulkFlow/Public/Run_Example2

```python step1_make_Union3.py```

```./pairV table.input table.input2```

```python step2_convert_to_fits.py```

```python step3_eig_reduce_n_at_a_time.py 1 100```
