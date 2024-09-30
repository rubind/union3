import pickle as pickle
import gzip
from scipy.special import erf
import numpy as np
import sys
import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from DavidsNM import miniNM_new
from FileRead import readcol
import os
multiprocessing.set_start_method("fork")
import time

UNION = os.environ["UNION"]

def randomexp():
    ind_choice = np.random.choice(np.arange(len(exp_approx_norm)), size = 1, replace = True, p = exp_approx_norm)[0]
    return np.random.normal()*exp_approx_width[ind_choice] + exp_approx_pos[ind_choice]

def normal_cdf(x, mu, sig):
    return 0.5*(1. + erf(  (x - mu)/(np.sqrt(2.) * sig)  ))




def make_a_SN(samp_ind, sn_ind, sn_pars):
    bad_fit = 1
    
    while bad_fit == 1:
        p_high_mass_eff = (1.9*(1 - fit_params["delta_h"][samp_ind])/(1 + 0.9*np.exp(0.95*np.log(10.)*stan_data["redshifts"][sn_ind])) + fit_params["delta_h"][samp_ind])*stan_data["p_high_mass"][sn_ind]
        beta_R = fit_params["beta_R_low"][samp_ind]*(1. - p_high_mass_eff) + fit_params["beta_R_high"][samp_ind]*p_high_mass_eff


        true_cB = np.random.normal()*fit_params["R_c_by_SN"][samp_ind][sn_ind] + fit_params["c_star_by_SN"][samp_ind][sn_ind]
        true_cR = randomexp()*fit_params["tau_c_by_SN"][samp_ind][sn_ind]
        true_x1 = np.random.normal()*fit_params["R_x1_by_SN"][samp_ind][sn_ind] + randomexp()*fit_params["tau_x1_by_SN"][samp_ind][sn_ind] + fit_params["x1_star_by_SN"][samp_ind][sn_ind]
        true_mB = fit_params["MB"][samp_ind][0] + fit_params["model_mu"][samp_ind][sn_ind] - fit_params["alpha"][samp_ind]*true_x1 + fit_params["beta_B"][samp_ind]*true_cB + beta_R*true_cR - fit_params["delta_0"][samp_ind]*p_high_mass_eff


        obs_cov_mat = np.zeros([3,3], dtype=np.float64)
        obs_cov_mat[0,0] = modelfn_var(P = sn_pars["mBmB"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)
        obs_cov_mat[1,1] = modelfn_var(P = sn_pars["x1x1"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)
        obs_cov_mat[2,2] = modelfn_var(P = sn_pars["cc"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)

        obs_cov_mat[0,1] = modelfn_var(P = sn_pars["mBx1"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)
        obs_cov_mat[1,0] = modelfn_var(P = sn_pars["mBx1"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)

        obs_cov_mat[0,2] = modelfn_var(P = sn_pars["mBc"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)
        obs_cov_mat[2,0] = modelfn_var(P = sn_pars["mBc"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)

        obs_cov_mat[1,2] = modelfn_var(P = sn_pars["x1c"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)
        obs_cov_mat[2,1] = modelfn_var(P = sn_pars["x1c"], mB0 = sn_pars["mB0"], mB = true_mB, c = true_cB + true_cR)

        cov_mat_unexpl_disp = fit_params["model_mBx1c_cov"][samp_ind][sn_ind] - stan_data["obs_mBx1c_cov"][sn_ind]

        obs_mBx1c = np.array([true_mB, true_x1, true_cB + true_cR]) + np.random.multivariate_normal([0., 0., 0.], cov_mat_unexpl_disp + obs_cov_mat)
        obs_mBx1c -= np.dot(stan_data["d_mBx1c_d_calib"][sn_ind], fit_params["calibs"][samp_ind]) # d_mBx1c_d_calib[i] * calibs
        obs_mag = obs_mBx1c[0] + stan_data["mobs_cut0"][sn_ind] + stan_data["mobs_cut1"][sn_ind]*obs_mBx1c[2]

        bad_fit = 0
        if np.abs(obs_mBx1c[1]) + np.sqrt(obs_cov_mat[1,1]) > 5:
            bad_fit = 1

        if np.abs(obs_mBx1c[2]) > 0.3:
            bad_fit = 1

        if bad_fit:
            print("Fit failed cuts!", np.abs(obs_mBx1c[1]) + np.sqrt(obs_cov_mat[1,1]), np.abs(obs_mBx1c[2]))
            
    obs_ddepth = np.random.normal()*fit_params["mobs_cut_sigmas"][samp_ind][stan_data["sample_list"][sn_ind] - 1]
    mag_limit = fit_params["mobs_cuts"][samp_ind][stan_data["sample_list"][sn_ind] - 1]
    

    if obs_mag + obs_ddepth < mag_limit:
        found = 1
    else:
        found = 0
        
    return dict(obs_mBx1c = obs_mBx1c,
                obs_mag = obs_mag,
                model_mu = fit_params["model_mu"][samp_ind][sn_ind],
                true_mBx1cBcR = [true_mB, true_x1, true_cB, true_cR],
                found = found)

def make_a_SN_conditional(samp_ind, sn_ind, sn_pars, needs_to_be_found = 1):
    found = 0
    close_x1c = 0
    closest_out_of_tries = 1000
    
    tries = 0
    
    while ((found == 0) or (close_x1c == 0)) and tries < 1e6:
        SN_dict = make_a_SN(samp_ind = samp_ind, sn_ind = sn_ind, sn_pars = sn_pars)

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
    sn_ind, samp_ind, sn_pars = both_inds
    
    return_dict = {}

    
    SN_dict = make_a_SN(samp_ind = samp_ind, sn_ind = sn_ind, sn_pars = sn_pars)

    return_dict["obs_mBx1c"] = SN_dict["obs_mBx1c"]
    return_dict["true_mBx1cBcR"] = SN_dict["true_mBx1cBcR"]
    return_dict["found"] = SN_dict["found"]
    return_dict["model_mu"] = SN_dict["model_mu"]
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


def modelfn_var(P, mB, c, mB0):
    return P[0] + P[1]*10**(0.8*(mB - mB0)) + P[2]*10**(0.8*c)

def chi2fn(P, passdata):
    [mB, c, mB0, var] = passdata[0]
    the_mod = modelfn_var(P, mB = mB, c = c, mB0 = mB0)
    pulls = (var - the_mod)/0.001

    return np.dot(pulls, pulls)


def get_cmat_interp_fns(snpath):
    [mB, c, mBmB, x1x1, cc, mBx1, mBc, x1c] = readcol(snpath + "/mBx1c_cov_by_mag_color.txt", 'ff,fff,fff')
    mB0 = np.mean(mB)

    assert mB0 > 0
    
    sn_pars = {"mB0": mB0}
    for var, var_name in [(mBmB, "mBmB"),
                          (x1x1, "x1x1"),
                          (cc, "cc"),
                          (mBx1, "mBx1"),
                          (mBc, "mBc"),
                          (x1c, "x1c")]:
        P, NA, NA = miniNM_new(ministart = [0.01, 0.01, 0.01], miniscale = [0.01, 0.01, 0.01],
                               chi2fn = chi2fn, passdata = [mB, c, mB0, var], compute_Cmat = False, verbose = False)
        
        sn_pars[var_name] = P
    return sn_pars


"""
sn_pars = get_cmat_interp_fns("/Users/rubind/Dropbox/Shared/Union3_Photometry/Union3_Union3_Submitted_2023-11_with_uncfit/Krisciunas/SN2000ca")

print(np.sqrt(modelfn_var(P = sn_pars["cc"], mB0 = sn_pars["mB0"], mB = 18., c = -0.2)))
print(np.sqrt(modelfn_var(P = sn_pars["cc"], mB0 = sn_pars["mB0"], mB = 18., c = 0.)))
print(np.sqrt(modelfn_var(P = sn_pars["cc"], mB0 = sn_pars["mB0"], mB = 18., c = 0.5)))
""" 


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
PPD["model_mu"] = []

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
        new_path = UNION + "/".join(the_data["snpaths"][sn_ind].split("/")[-2:])
        
        sn_pars = get_cmat_interp_fns(new_path)
        all_return_dicts = pool.map(single_SN_fn, zip([sn_ind]*n_samples, np.arange(n_samples), [sn_pars]*n_samples))

        for return_dict in all_return_dicts:
            for key in return_dict:
                PPD[key][-1].append(return_dict[key])
        
        print("Found probability", np.mean(PPD["found"][-1]))


print("Dumping pickle", time.asctime())
pickle.dump(PPD, open("PPD.pickle", 'wb'))

print("Done", time.asctime())

for key in PPD:
    print(key, PPD[key].shape)

