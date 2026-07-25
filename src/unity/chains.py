"""Helpers for reading the (potentially very wide) mcmc_samples.parquet chain files.

All-latents runs produce ~100k columns. These files are fine to read as long as
you select columns instead of loading everything: opening the file costs ~1s and
~0.5 GB (parquet footer metadata, irreducible at this column count), after which
any column subset streams quickly. A full `pl.read_parquet` on a 9 GB file needs
~2x the file size in RAM — use `read_samples` with `columns=` or `bases=` instead.
"""

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


def read_samples(
    path: Path | str,
    columns: list[str] | None = None,
    bases: list[str] | str | None = None,
) -> pl.DataFrame:
    """Read a column subset of an mcmc_samples.parquet file.

    `columns` selects exact column names. `bases` selects every index of the
    given base parameter name(s), e.g. bases="true_cR" returns `true_cR[1]`
    through `true_cR[2085]` (a bare scalar with no bracket index also matches).
    Both may be given; the union is returned, in file order.
    """
    pf = pq.ParquetFile(path)
    if columns is None and bases is None:
        return pl.from_arrow(pf.read())

    if isinstance(bases, str):
        bases = [bases]
    wanted = set(columns or [])
    prefixes = tuple(f"{b}[" for b in bases or [])
    selected = [
        name
        for name in pf.schema_arrow.names
        if name in wanted or name in (bases or []) or name.startswith(prefixes)
    ]
    selected_set = set(selected)
    missing = wanted - selected_set
    missing.update(b for b in bases or [] if b not in selected_set and not any(n.startswith(f"{b}[") for n in selected))
    if missing:
        raise KeyError(f"Columns not present in {path}: {sorted(missing)}")
    return pl.from_arrow(pf.read(columns=selected))


def chain_column_names(path: Path | str) -> list[str]:
    """List the column names of a chain file without reading any data."""
    return pq.ParquetFile(path).schema_arrow.names
