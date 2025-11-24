from collections import defaultdict
from io import StringIO
import re
from pydantic import computed_field
from pydantic_settings import BaseSettings
from pathlib import Path
from union3 import logger
import polars as pl
from astropy.coordinates import SkyCoord
from astropy import units as u


# Create a count decorator to count the number of invocations of a function
def count(func):
    def wrapper(*args, **kwargs):
        wrapper.count += 1  # type: ignore
        return func(*args, **kwargs)

    wrapper.count = 0  # type: ignore
    return wrapper


class LCFitExtractionConfig(BaseSettings):
    union3_dir: str = "tmp_dropbox"

    @property
    def union3_path(self) -> Path:
        return Path(__file__).parent / self.union3_dir

    @computed_field
    @property
    def output_dir(self) -> Path:
        return Path(__file__).parents[2] / "union3/data/resources/supernova_lc_fits"


def process_survey(survey_path: Path, config: LCFitExtractionConfig) -> int:
    logger.info(f"Processing directory: {survey_path.name}")

    # There should be a survey_name_v1.txt file in the survey directory which contains the
    # list of supernovae which passed light curve fitting cuts.
    pass_file = survey_path.parent / f"{survey_path.name}_v1.txt"
    assert pass_file.exists(), f"Pass file {pass_file} does not exist."
    passed_sn_dirs = {line.strip().split("/")[-1] for line in pass_file.read_text().strip().splitlines()}

    data = []
    for sn_dir in sorted(survey_path.iterdir()):
        if sn_dir.is_dir():
            sn_data = process_supernova(sn_dir, survey_path.name)
            passed = sn_dir.name in passed_sn_dirs
            if sn_data is not None:
                data.append(sn_data | {"lcfit_passed": passed})

    if not data:
        logger.warning(f"No data found for survey {survey_path.name}")
        return 0

    df = pl.concat([pl.DataFrame(item) for item in data], how="diagonal_relaxed")
    output_file = config.output_dir / f"{survey_path.name}.parquet"
    df.write_parquet(output_file)
    logger.info(f"Written data to {output_file}")
    return len(data)


@count
def process_supernova(sn_path: Path, survey: str):
    sn_name = sn_path.name
    logger.info(f"\tProcessing SN: {sn_name}")
    try:
        data = (
            load_light_file(sn_path)
            | load_results(sn_path)
            | load_derivatives(sn_path, "model_deriv.dat")
            | load_derivatives(sn_path, "result_deriv.dat")
            | {"name": sn_name, "survey": survey}
        )
        if len(data) <= 5:  # name, survey + few more
            logger.warning(f"\tNo useful data found for {sn_name} in {survey}, skipping.")
            return None
        return data
    except Exception as e:
        logger.opt(exception=e).error(f"\tError processing {sn_name}: {e}")
        raise


def maybe_float(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def load_light_file(sn_path: Path):
    light_file = sn_path / "lightfile"
    if not light_file.exists():
        return {}
    content = light_file.read_text().strip().splitlines()
    data = {}
    for line in content:
        data |= parse_bullshit_line(line)
    return data


def parse_bullshit_line(line: str) -> dict:
    if line.startswith("#") or not line.strip():
        return {}
    key, *values = line.split()
    key = key.strip().lower()
    data = {}
    if len(values) == 0:
        return {}
    if len(values) >= 1:
        data[key] = maybe_float(values[0])
    if len(values) == 2:
        err = maybe_float(values[1])
        if isinstance(err, float) and err > 0:
            data[f"{key}_err"] = err
    elif len(values) == 3:
        lower_err = maybe_float(values[1])
        upper_err = maybe_float(values[2])
        if isinstance(lower_err, float):
            data[f"{key}_err_lower"] = lower_err
        if isinstance(upper_err, float):
            data[f"{key}_err_upper"] = upper_err
    val = data[key]
    if not isinstance(val, float) and ":" in str(val):
        if key == "ra":
            coord = SkyCoord(ra=val, dec="0", unit=(u.deg, u.deg))  # type: ignore
            data[key] = coord.ra.deg  # type: ignore
        elif key == "dec":
            coord = SkyCoord(ra="0", dec=val, unit=(u.deg, u.deg))  # type: ignore
            data[key] = coord.dec.deg  # type: ignore
        else:
            raise ValueError(f"Unexpected key {key} for sexagesimal format")
    return data


def load_results(sn_path: Path):
    results_file = sn_path / "result_salt2.dat"
    if not results_file.exists():
        return {}
    content = results_file.read_text().strip().splitlines()
    data = {}
    for line in content[2:]:
        data |= parse_bullshit_line(line)
    return data


def load_derivatives(sn_path: Path, filename: str):
    derivatives_file = sn_path / filename
    if not derivatives_file.exists():
        return {}
    content = re.sub(r" +", " ", derivatives_file.read_text().strip())
    basename = derivatives_file.stem
    df = pl.read_csv(StringIO(content), separator=" ", has_header=True, skip_lines=1)
    df = df[[s.name for s in df if not (s.null_count() == df.height)]]
    df = (
        df.with_columns(
            prefix=f"{basename}_"
            + pl.col("#Parameter")
            + "_"
            + pl.col("MagSys|Instrument|Band").str.replace(r"All\|All\|All", "")
            + "_"
            + pl.col("Phase")
        )
        .drop("#Parameter", "MagSys|Instrument|Band", "Phase")
        .unpivot(index="prefix")
        .with_columns(name=(pl.col("prefix") + "_" + pl.col("variable")).str.replace_all(r"__", r"_"), tmp=1)
        .select("name", "value", "tmp")
        .pivot(on="name", values="value", index="tmp")
        .drop("tmp")
    )
    result = df.to_dicts()[0]
    return {k: maybe_float(v) for k, v in result.items()}


def read_lcfit_failures(failures_file: Path) -> dict[str, set[str]]:
    """Returns a dictionary mapping survey names to lists of failed supernova names."""
    assert failures_file.exists(), f"LCFit failure file {failures_file} does not exist."
    results = defaultdict(set)
    lines = [x.split("/") for x in failures_file.read_text().strip().splitlines()]
    for survey, sn_name, *_ in lines:
        results[survey].add(sn_name)
    return results


def main():
    config = LCFitExtractionConfig()

    logger.info(f"LCFit Extraction with params: {config.model_dump_json(indent=2)}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    num_snia = {}
    for directory in sorted(config.union3_path.iterdir()):
        if directory.is_dir():
            num_snia[directory.stem] = process_survey(directory, config)

    logger.info("Processed supernova counts by survey:")
    for survey, count in num_snia.items():
        logger.info(f"\t{survey}: {count}")
    logger.info(f"Processed {process_supernova.count} supernovae.")  # type: ignore


if __name__ == "__main__":
    main()
