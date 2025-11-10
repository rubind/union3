The original data for this repository was provided as a tarball at https://www.dropbox.com/scl/fi/amiyz4504aoemb6gchtqo/Union31.tar.gz, and this contained the light curve data, diagnostic plots, logs, and many other things we do not need to fit cosmology.

This directory is how I have processed this tarball into something a bit easier to handle. To run this, you will need to download and extract the above tarball, and set the local config to point toward it, either via cli, env var, or simply changing the code default.

By running the main.py file in this directory, you should regenerate the summary parquet files in the top level `data` directory, and this will include information from the lightfile, the salt2 results file, and the salt2 result derivative file.