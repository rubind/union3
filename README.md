# union3
Union3/UNITY1.5 repo

```export UNITY=path/to/union3```

```export UNION=path/to/lcfits```

For the LC fits:

```python parse_nearby.py paramfile_default.txt```

For magnitude systems and instruments: edit the original_* files in $PATHMODEL. Then run python_code/shiftfilters.py if you want to be able to use snfit (as opposed to SALT3.py).

From the Union3 directory, run:

```python $PATHMODEL/python_code/tmpfit.py 0 4 1```
