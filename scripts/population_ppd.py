import pickle as pickle
import gzip
from scipy.special import erf
import numpy as np
import sys
import tqdm
import matplotlib.pyplot as plt
import multiprocessing
multiprocessing.set_start_method("fork")
import time


def randomexp():
    ind_choice = np.random.choice(np.arange(len(exp_approx_norm)), size = 1, replace = True, p = exp_approx_norm)[0]
    return np.random.normal()*exp_approx_width[ind_choice] + exp_approx_pos[ind_choice]

def normal_cdf(x, mu, sig):
    return 0.5*(1. + erf(  (x - mu)/(np.sqrt(2.) * sig)  ))

def make_a_SN(samp_ind, sn_ind):
    p_high_mass_eff = (1.9*(1 - fit_params["delta_h"][samp_ind])/(1 + 0.9*np.exp(0.95*np.log(10.)*stan_data["redshifts"][sn_ind])) + fit_params["delta_h"][samp_ind])*stan_data["p_high_mass"][sn_ind]
    beta_R = fit_params["beta_R_low"][samp_ind]*(1. - p_high_mass_eff) + fit_params["beta_R_high"][samp_ind]*p_high_mass_eff
    
    
    true_cB = np.random.normal()*fit_params["R_c_by_SN"][samp_ind][sn_ind] + fit_params["c_star_by_SN"][samp_ind][sn_ind]
    true_cR = randomexp()*fit_params["tau_c_by_SN"][samp_ind][sn_ind]
    true_x1 = np.random.normal()*fit_params["R_x1_by_SN"][samp_ind][sn_ind] + randomexp()*fit_params["tau_x1_by_SN"][samp_ind][sn_ind] + fit_params["x1_star_by_SN"][samp_ind][sn_ind]
    true_mB = fit_params["MB"][samp_ind][0] + fit_params["model_mu"][samp_ind][sn_ind] - fit_params["alpha"][samp_ind]*true_x1 + fit_params["beta_B"][samp_ind]*true_cB + beta_R*true_cR - fit_params["delta_0"][samp_ind]*p_high_mass_eff

    obs_mBx1c = np.array([true_mB, true_x1, true_cB + true_cR]) + np.random.multivariate_normal([0., 0., 0.], fit_params["model_mBx1c_cov"][samp_ind][sn_ind])
    obs_mag = obs_mBx1c[0] + stan_data["mobs_cut0"][sn_ind] + stan_data["mobs_cut1"][sn_ind]*obs_mBx1c[2]


    obs_ddepth = np.random.normal()*fit_params["mobs_cut_sigmas"][samp_ind][stan_data["sample_list"][sn_ind] - 1]
    mag_limit = fit_params["mobs_cuts"][samp_ind][stan_data["sample_list"][sn_ind] - 1]
    

    if obs_mag + obs_ddepth < mag_limit:
        found = 1
    else:
        found = 0
        
    return dict(obs_mBx1c = obs_mBx1c,
                obs_mag = obs_mag,
                true_mBx1cBcR = [true_mB, true_x1, true_cB, true_cR],
                found = found)

def make_a_SN_conditional(samp_ind, sn_ind, needs_to_be_found = 1):
    found = 0
    close_x1c = 0
    closest_out_of_tries = 1000
    
    tries = 0
    
    while ((found == 0) or (close_x1c == 0)) and tries < 1e6:
        SN_dict = make_a_SN(samp_ind = samp_ind, sn_ind = sn_ind)

        if needs_to_be_found:
            found = SN_dict["found"]
        else:
            found = 1
        
        x1_miss = stan_data["obs_mBx1c"][sn_ind][1] - SN_dict["obs_mBx1c"][1]
        c_miss = stan_data["obs_mBx1c"][sn_ind][2] - SN_dict["obs_mBx1c"][2]
        total_miss = (x1_miss/0.1)**2. + (c_miss/0.01)**2
        
        close_x1c = total_miss < 0.5
        tries += 1

        if found:
            if total_miss < closest_out_of_tries:
                closest_out_of_tries = total_miss
                best_dict = dict(**SN_dict)
            
    return best_dict


def single_SN_fn(both_inds):
    sn_ind, samp_ind = both_inds
    
    return_dict = {}

    
    SN_dict = make_a_SN(samp_ind = samp_ind, sn_ind = sn_ind)

    return_dict["obs_mBx1c"] = SN_dict["obs_mBx1c"]
    return_dict["true_mBx1cBcR"] = SN_dict["true_mBx1cBcR"]
    return_dict["found"] = SN_dict["found"]
    return_dict["mB_unc"] = np.sqrt(fit_params["model_mBx1c_cov"][samp_ind][sn_ind][0,0])
    return_dict["x1_unc"] = np.sqrt(fit_params["model_mBx1c_cov"][samp_ind][sn_ind][1,1])
    return_dict["c_unc"] = np.sqrt(fit_params["model_mBx1c_cov"][samp_ind][sn_ind][2,2])

    

    """
    if samp_ind % 100 == 0:
        SN_dict = make_a_SN_conditional(samp_ind = samp_ind, sn_ind = sn_ind, needs_to_be_found = 1)
        return_dict["obs_mBx1c|foundobsx1c"] = SN_dict["obs_mBx1c"]
        return_dict["true_mBx1cBcR|foundobsx1c"] = SN_dict["true_mBx1cBcR"]
        
        SN_dict = make_a_SN_conditional(samp_ind = samp_ind, sn_ind = sn_ind, needs_to_be_found = 0)
        return_dict["obs_mBx1c|obsx1c"] = SN_dict["obs_mBx1c"]
        return_dict["true_mBx1cBcR|obsx1c"] = SN_dict["true_mBx1cBcR"]
    """
    return return_dict

input_fl = sys.argv[1]
sample_fl = sys.argv[2]

exp_approx_norm = np.array([0.15038540936467037, 0.2993904768085472, 0.364279051173158, 0.18594506265362443])
exp_approx_pos = np.array([0.10329973984501734, 0.41080906196995237, 1.083137332416308, 2.427349566890827])
exp_approx_width = np.array([0.06596419371844692, 0.1910889454034621, 0.45516250820784515, 1.0637414822809306])


(the_data, stan_data, params) = pickle.load(gzip.open(input_fl, 'rb'))
fit_params = pickle.load(gzip.open(sample_fl, 'rb'))

for key in the_data:
    print("the_data", key)

for key in fit_params:
    print("fit_params", key)
    
    

assert stan_data["MB_by_sample"] == 0
assert stan_data["do_host_mass"] == 1


PPD = {}

PPD["obs_mBx1c"] = []
PPD["found"] = []
PPD["true_mBx1cBcR"] = []
PPD["mB_unc"] = []
PPD["x1_unc"] = []
PPD["c_unc"] =	[]

"""
PPD["obs_mBx1c|foundobsx1c"] = []
PPD["true_mBx1cBcR|foundobsx1c"] = []

PPD["obs_mBx1c|obsx1c"] = []
PPD["true_mBx1cBcR|obsx1c"] = []
"""

pool = multiprocessing.Pool(processes = 10)

n_samples = len(fit_params["alpha"])

for sn_ind in tqdm.trange(stan_data["n_sne"]):
    in_minus_out = np.median(fit_params["inl_loglike_by_SN"][:, sn_ind]) - np.median(fit_params["outl_loglike_by_SN"][:, sn_ind])
    print(the_data["snpaths"][sn_ind], in_minus_out)
    
    for key in PPD:
        PPD[key].append([])
        
    if in_minus_out > 0:
        all_return_dicts = pool.map(single_SN_fn, zip([sn_ind]*n_samples, np.arange(n_samples)))

        for return_dict in all_return_dicts:
            for key in return_dict:
                PPD[key][-1].append(return_dict[key])
        
        print("Found probability", np.mean(PPD["found"][-1]))


print("Dumping pickle", time.asctime())
pickle.dump(PPD, open("PPD.pickle", 'wb'))

print("Done", time.asctime())

for key in PPD:
    print(key, PPD[key].shape)

