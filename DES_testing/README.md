# DES SNANA-to-SALT Testing

This directory contains scripts and intermediate products used to:

1. convert SNANA-format SN Ia light curves into the per-supernova SALT
   light-curve format used by `salt2_union`;
2. fit those light curves with SALT3;
3. compare the resulting Python/SNCosmo fits with SNANA fits and simulation
   truth; and
4. optionally convert SNANA `FITRES` tables into UNITY-compatible input
   directories.

The directory is an analysis workspace rather than a self-contained software
package. Several scripts assume David Rubin's `PATHMODEL`/UNITY environment and
the University of Hawaii Koa Slurm cluster.

## Environment

The following environment variables are used:

```bash
export PATHMODEL=/path/to/salt2_union
export UNITY=/path/to/union3
```

`PATHMODEL` must contain:

- `python_code/SALT3.py`
- `python_code/slurmfit.py`
- `python_code/FileRead.py`
- `SALT3.DES5YR/`
- `Instruments/DECam/`

On the Mac used to develop this directory, the suitable Python interpreter is:

```bash
/opt/anaconda3/bin/python
```

It has the required NumPy, SciPy, pandas, Astropy, Matplotlib, tqdm, sncosmo,
and `FileRead` dependencies. The unqualified `python` command may point to an
old Python installation, so check it before use.

## Primary workflow: SNANA light curves to SALT fits

### 1. Extract the SNANA files

`DATADESSIM.zip` contains ten realizations, each with a
`HEAD.FITS.gz`/`PHOT.FITS.gz` pair:

```bash
unzip DATADESSIM.zip
```

The ZIP archive currently passes an integrity check:

```bash
unzip -tq DATADESSIM.zip
```

### 2. Convert the light curves

Run:

```bash
/opt/anaconda3/bin/python DES_convert_to_lcs.py \
  DATADESSIM/*/*HEAD.FITS.gz
```

For each SN with more than five observations,
`DES_convert_to_lcs.py` creates a directory such as:

```text
DATADESSIM/<realization>/DES_sim_Deep/<SNID>/
DATADESSIM/<realization>/DES_sim_Shallow/<SNID>/
```

Each SN directory contains:

```text
lightfile
lc2fit_DES-g.dat
lc2fit_DES-r.dat
lc2fit_DES-i.dat
lc2fit_DES-z.dat
```

The converter maps SNANA `FLUXCAL` measurements to SALT files with zeropoint
27.5 and maps the DES bands to the `DECam_g`, `DECam_r`, `DECam_i`, and
`DECam_z` passbands.

The script also writes `DATADESSIM/slurmfit.sh`. That generated script contains
Koa-specific paths beginning with `/home/drubin/koa_scratch/`; edit or
regenerate those paths when running elsewhere.

### 3. Fit with SALT3

On Koa, the intended entry point is:

```bash
sbatch DATADESSIM/slurmfit.sh
```

For each realization, this invokes:

```bash
python "$PATHMODEL/python_code/slurmfit.py" \
  10 dontsort \
  --salt2_version SALT3.DES5YR \
  --wave 3500 8000
```

`slurmfit.py` finds `*/*/lightfile`, groups ten SNe per job, and submits jobs
that run approximately:

```bash
cd DATADESSIM/<realization>/DES_sim_<depth>/<SNID>
python "$PATHMODEL/python_code/SALT3.py" lc2*dat \
  --salt2_version SALT3.DES5YR \
  --wave 3500 8000
```

The main per-SN result is `result_salt2.dat`; the fitter also writes diagnostic
and derivative files.

There is no `sbatch` command on a normal local Mac. A local runner would need
to call `SALT3.py` directly, preferably with controlled parallelism.

### 4. Assemble the fit summary

The comparison code expects:

```text
all_LC_fits.txt
```

with at least these columns:

```text
path mB x1 c dmB dx1 dc z_helio
```

The current file also contains wavelength-derivative columns. The script that
originally assembled this table from the individual SALT results is not
present in this directory. A fresh end-to-end run therefore requires restoring
that script or writing a replacement.

### 5. Compare Python and SNANA fits

The comparison requires:

- `all_LC_fits.txt`, containing the Python/SNCosmo SALT fits; and
- `ALL_SIMS_FITOPT000.FITRES`, containing the SNANA fits and simulation truth.

Run:

```bash
MPLCONFIGDIR=/private/tmp/des-mpl \
  /opt/anaconda3/bin/python compare_SNANA_python_LC_fits.py
```

This matches objects using realization and SNID, compares `mB`, `x1`, `c`,
their uncertainties and pulls, and writes:

```text
LC_check.pdf
```

`check_LC_fits.py` is an older comparison against truth read directly from the
original ten-realization `HEAD.FITS` files.

## Newer 25-realization conversion

`Unite_convert_to_lcs.py` is a newer converter used for input directories
matching `*SIMDES_IA*`. It writes light curves below `sim_converted/` and was
used for the 25-realization products represented by the current
`all_LC_fits.txt` and `ALL_SIMS_FITOPT000.FITRES`.

It is not currently runnable as-is because:

- the source `*SIMDES_IA*` directories are not present here;
- it deletes and recreates `sim_converted/`; and
- its final Slurm-script block still writes to `DATADESSIM/slurmfit.sh`.

`Unite_convert_filts.py` constructs a separate `Unite_PATHMODEL` using a
`SALT3.UNITE` model and DES filters. It assumes the source `filters/` and
`SALT3.UNITE/` directories are present and writes Koa-specific paths.

## FITRES-to-UNITY workflow

`convert_to_sim_LC_fits.py` does **not** fit light curves. Instead, it converts
an existing SNANA `FITRES` table into UNITY-compatible directories containing
`result_salt2.dat`, `lightfile`, covariance information, sample lists, and
UNITY 1.7/1.8 parameter files.

Its interface is:

```bash
/opt/anaconda3/bin/python convert_to_sim_LC_fits.py \
  ALL_SIMS_FITOPT000.FITRES
```

**Warning:** this script begins by deleting and recreating `sim_convert/`.
Back up or rename an existing `sim_convert/` directory before running it.
Generated `run.sh` files also contain Koa-specific absolute paths.

## Other scripts

- `convert_SNANA_LC_fit.py` converts older per-realization SNANA FITRES files
  below `SNANA_LC_FIT/` into `SNANA_LC.txt`.
- `check_bluer_4000.py` plots the effect of the wavelength-derivative columns
  in `all_LC_fits.txt`.
- `set_up_UNITY.py` creates and immediately submits older UNITY jobs for
  directories supplied on the command line. It deletes an existing `UNITY/`
  subdirectory in each target.
- `corner_plots.py` and `look_at_beta_R.py` analyze saved UNITY samples.
- Files ending in `~` are editor backups and should not be treated as the
  canonical scripts.

## Existing products and cautions

- `DATADESSIM.zip` is the intact ten-realization SNANA input archive.
- `DATADESSIM_SALT_LC.tar.gz` is truncated and fails `gzip -t`; do not rely on
  it as the only copy of converted light curves.
- `all_LC_fits.txt` currently contains 54,463 fitted light curves from 25
  realizations.
- `ALL_SIMS_FITOPT000.FITRES` currently contains 49,131 SNANA fit rows from 25
  realizations.
- `LC_check.pdf` is the existing Python-versus-SNANA comparison.

Before rerunning any script, inspect it for `rm -fr`, `sbatch`, and
`/home/drubin/koa_scratch` because several files are job-generation scripts
with destructive setup steps and machine-specific paths.
