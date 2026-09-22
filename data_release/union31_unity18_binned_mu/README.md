# Union3.1 / UNITY 1.8 binned distance moduli

Public data release: binned distance moduli and their covariance, for use in external cosmology
fits.

| file | contents |
|---|---|
| `mu_binned.ecsv` | bin redshifts, binned distance moduli, per-bin uncertainties, and the full 22x22 covariance. Plain text; read with `astropy.table.Table.read`. |
| `mu_mat.fits` | the same content packed into a single 23x23 float64 image, in the layout used by past Union releases. |

Most users want `mu_binned.ecsv`. `mu_mat.fits` exists so that consumers written against earlier
Union releases keep working unchanged.

## mu_mat.fits layout

```
[0, 0]   = 0
[0, 1:]  = bin redshifts
[1:, 0]  = binned distance modulus per bin
[1:, 1:] = INVERSE covariance
```

Row and column zero are metadata, not part of the matrix. The inner block is the inverse
covariance, for backward compatibility with past Union releases; `mu_binned.ecsv` carries the
covariance itself.

## How to use these

The distance moduli are residuals relative to `FlatLambdaCDM(H0=70, Om0=0.3)`. Add that fiducial
back at the bin redshifts to recover absolute distance moduli. That step is exact, since each
interpolation basis function is 1 at its own node and 0 at the others.

Fit with a free magnitude offset (scriptM). The mean of the bins is held near zero by the z=0
anchor of the interpolation basis. That is a normalisation, not a measurement of absolute scale,
which would require calibrator distances or a sound-horizon prior. A fit that holds the offset
fixed will report spuriously tight constraints.

The bins themselves are free parameters with no cosmological relation imposed between them: no
expansion history is fit in the run that produces them. The fiducial above does set the shape
within a bin, which matters only at the level of a few tenths of a sigma in the highest-redshift
bins, where the bins are widest and the statistical errors largest. The repository README has the
numbers.

## Provenance

| | |
|---|---|
| config | `src/unity/configs/union31_unity18_published_binnedMu.yml` |
| supernovae / bins | 2085 / 22 |
| chains x draws | 4 x 5000 after 1000 warmup = 20000 |
| sampler | NumPyro NUTS |
| divergences | 0 |
| max split-Rhat | 1.0077 overall, 1.0016 across the 22 binned distance moduli |

The commit and sampling seed are recorded in the files themselves, in the FITS header and in the
ECSV metadata.

## Regenerating them

From the repository root, with the environment installed:

```bash
uv run unity --base union31_unity18_published_binnedMu.yml
```

That writes `mu_mat.fits`, `mu_binned.ecsv` and the chains into the run's output directory; the two
release files are then copied here. The run takes about 2.5 hours on four cores. Runs are blinded
by default, and the flags that disable blinding are typed on the command line rather than stored in
any file, so the command above reproduces the pipeline but not the unblinded product.

The released numbers are medians and a sample covariance over 20000 draws, so an independent run
with a different seed reproduces them to Monte Carlo error rather than bit-for-bit.

## Chains

The full posterior chains are not included here. They are available on request: 20000 draws of 66
parameters, which is the global parameter set plus the 22 binned distance moduli and the sampler
diagnostics. No per-supernova latent parameters are saved.
