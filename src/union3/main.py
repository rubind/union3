from pathlib import Path
import pickle
import multiprocessing

import stan
import gzip

from union3.utils.init_fn import init_fn
from union3.utils.augment import add_zbins
from union3.utils.blind import blind


def main():
    multiprocessing.set_start_method("fork")

    here = Path(__file__).parent
    inputfl = (
        here
        / "data"
        / "inputs_Amanullah10_CNIa02_CSP_CalanTololo_CfA1_CfA2_CfA3_CfA4_DES3_Deep_DES3_Shallow_ESSENCE_Foundation_LOSS_MCT_NB99_Pan-STARRS_Riess07_SDSS_SNLS_SuzukiRubin_Tonry03_LSQ+LCO_LSQ_knop03_Krisciunas.pickle"
    )
    print(
        "cosmo_model: 1 for Om, 2 for binned mu, 3 for Omega_m-w, 4 for q0-j0, 5 for Omega_m-w0-wa, 6 for binned mu with comoving interpolation"
    )
    # cosmo_model = int(sys.argv[2])
    cosmo_model = 1

    (the_data, stan_data, params) = pickle.load(gzip.open(inputfl, "rb"))

    stan_data = add_zbins(stan_data, cosmo_model)

    print("nzadd ", stan_data["nzadd"])

    if stan_data["do_blind"]:
        stan_data, the_data = blind(stan_data, the_data, params)
    else:
        print("Not Blinding!")

    print("Running...")

    stan_file = here / "models" / "stan_code_H0_1.8.txt"
    stan_model = stan_file.read_text()

    posterior = stan.build(stan_model, data=stan_data)
    fit = posterior.sample(
        num_samples=10,  # params["iter"],
        num_warmup=10,
        num_chains=params["chains"],
        # n_jobs=params["n_jobs"],
        # refresh=10,
        init=[init_fn(stan_data, the_data) for _ in range(params["chains"])],
        # sample_file=params["sample_file"],
    )

    samples = fit.to_frame()
    samples.to_parquet("samples.parquet")


# try:
#     fit_params = helper_functions.filter_fit_params(
#         fit_params, "MB", params["chains"], params["iter"] / 2
#     )  # burns the first half of the chain, so iter/2
# except:
#     print("Couldn't filter bad chains! One or more chains may be bad!")

# # summarize_parameters(fit_params)

# helper_functions.write_latent_variables(the_data=the_data, stan_data=stan_data, fit_params=fit_params)


# del_keys = []
# for key in fit_params:
#     sh = np.array(fit_params[key].shape)

#     if np.any(sh[1:] > params["max_params_to_save"]):
#         print(key, " is too big to save!", sh)
#         del_keys.append(key)

# print("del_keys", del_keys)
# for key in del_keys:
#     del fit_params[key]


# pickle.dump(fit_params, gzip.open("samples.pickle", "wb"))


# print("I hope you have a log file:")

# try:
#     print(fit.stansummary(digits_summary=5))
# except:
#     print("Couldn't print fit! Something is very wrong!")

if __name__ == "__main__":
    main()
