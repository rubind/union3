from matplotlib import use
use("PDF")
import pickle
from numpy import *
import pystan
import sys
import os
from string import strip
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
import helper_functions
from scipy.interpolate import interp1d


################################################# Get the SALT data ###################################################


def read_data(params):

    the_data = {"mB_list": array([], dtype=float64),
                "x1_list": array([], dtype=float64),
                "c_list": array([], dtype=float64),
                "mBx1c_cov_list": zeros([0,3,3], dtype=float64),
                "z_CMB_list": array([], dtype=float64),
                "z_helio_list": array([], dtype=float64),
                "sample_list": array([], dtype=int32), # SN sample, from 0 to N_samples - 1
                "sample_names": [], # For storing sample names
                "mag_cut_list": array([], dtype=float64),
                "mag_cut_disp_list": array([], dtype=float64), # Dispersion on magnitude cut
                "calib_names": [], # Name of each systematic uncertainty
                "mass": [], # Host mass
                "mass_err": [], # Host-mass uncertainty
                "snpaths": [], # Paths to LC fits. Stored for future reference.
                "efflambs": {}, # Filter wavelengths
                "d_mBx1c_dcalib_list": zeros([3000,3,500], dtype=float64), # This is an inefficient way to do this, but this is initialized to fixed size, then trimmed later.
            }


    current_sn_ind = 0
    
    filenamelist = params["filenamelist"]
    assert len(filenamelist) > 1, "This code requires > 1 input file!"

    f_read = open("sn_input.txt", 'w')
    f_read.write("#SN\tRA\tDEC\tZHEL\tZCMB\tPASS\n") # List of SNe that pass all cuts

    for current_sample, directory in enumerate(filenamelist):
        the_data["sample_names"].append(directory)

        f = open(directory)
        snpaths = clean_lines(f.read().split('\n'))
        f.close()
        
        for snpath in snpaths:
            this_redshift = read_param(snpath + "/result_salt2.dat", "Redshift")
            this_redshift_cmb = read_param(snpath + "/lightfile", "z_cmb")
            this_redshift_helio = read_param(snpath + "/lightfile", "z_heliocentric")
            this_RA = read_param(snpath + "/lightfile", "RA")
            this_DEC = read_param(snpath + "/lightfile", "DEC")
            this_firstphase = read_param(snpath + "/result_salt2.dat", "FirstPhase")
            this_lastphase = read_param(snpath + "/result_salt2.dat", "LastPhase")
            this_colorerr = read_param(snpath + "/result_salt2.dat", "Color", ind = 2)
            this_x1 = read_param(snpath + "/result_salt2.dat", "X1", ind = 1)
            this_x1_err = read_param(snpath + "/result_salt2.dat", "X1", ind = 2)
            try:
                this_check = read_param(snpath + "/result_deriv.dat", "Check", ind = 4)
            except:
                this_check = 1000

            if this_x1_err == None:
                this_x1 = 100.
                this_x1_err = 100.

            weird_sn = read_param(params["weird_sn_list"], snpath.split("/")[-1])
            print "weird_sn ", snpath, weird_sn
            
            
            okay_to_add = [this_redshift >= params["min_redshift"],
                           this_redshift <= params["max_redshift"],
                           this_firstphase <= params["max_firstphase"],
                           this_lastphase >= params["min_lastphase"],
                           this_colorerr < params["max_color_uncertainty"],
                           weird_sn == None, abs(log(this_check)) < 0.1, abs(this_x1) + this_x1_err < 5]
            okay_names = ["min_z", "max_z", "first_p", "last_p", "colorerr", "weirdsn", "converge", "x1"]

            f_read.write('\t'.join([
                "/".join(snpath.split("/")[-2:]),
                str(this_RA),
                str(this_DEC),
                str(this_redshift_helio),
                str(this_redshift_cmb),
                str(all(okay_to_add))]
                               ) + '\n')


            if all(okay_to_add):

                the_data["snpaths"].append(snpath)
                the_data["z_CMB_list"] = append(the_data["z_CMB_list"], this_redshift
                                                )


                the_data["mB_list"] = append(the_data["mB_list"],
                                             read_param(snpath + "/result_salt2.dat", "RestFrameMag_0_B"))
                the_data["c_list"] = append(the_data["c_list"],
                                            read_param(snpath + "/result_salt2.dat", "Color"))
                the_data["x1_list"] = append(the_data["x1_list"],
                                             read_param(snpath + "/result_salt2.dat", "X1"))


                the_data["sample_list"] = append(the_data["sample_list"], current_sample)

                this_mass = read_param(snpath + "/lightfile", "Mass", ind = 1)
                this_mass_err = sqrt(abs(
                    read_param(snpath + "/lightfile", "Mass", ind = 2)*read_param(snpath + "/lightfile", "Mass", ind = 3)
                    ))

                if this_mass == None or this_mass < 1 or this_mass_err == 0. or isinf(this_mass_err):
                    if the_data["z_CMB_list"][-1] > 0.1:
                        the_data["mass"].append(10.)
                        the_data["mass_err"].append(1.)
                    else:
                        the_data["mass"].append(11.)
                        the_data["mass_err"].append(1.)
                else:
                    the_data["mass"].append(this_mass)
                    the_data["mass_err"].append(this_mass_err)


                # First term from SALT, second term from 300 km/s, third term lensing (may be overestimated)
                mBmB = read_param(snpath + "/result_salt2.dat", "RestFrameMag_0_B", ind = 2)**2. + (params["pec_vel_disp"]/the_data["z_CMB_list"][-1]*5./log(10.))**2. + (params["lensing_disp"]*the_data["z_CMB_list"][-1])**2.
                mBx1 = read_param(snpath + "/result_salt2.dat", "CovRestFrameMag_0_BX1")
                mBc = read_param(snpath + "/result_salt2.dat", "CovColorRestFrameMag_0_B")
                x1x1 = read_param(snpath + "/result_salt2.dat", "CovX1X1")
                x1c = read_param(snpath + "/result_salt2.dat", "CovColorX1")
                cc = read_param(snpath + "/result_salt2.dat", "CovColorColor")

                h_resid = (the_data["mB_list"][-1] - - 19.1 + 0.13*the_data["x1_list"][-1] - 3.*the_data["c_list"][-1]) - (5*log10(the_data["z_CMB_list"][-1]*(1. + the_data["z_CMB_list"][-1])) + 42.9)
                if abs(h_resid) > 1.5 or (the_data["c_list"][-1] > 1) or (the_data["c_list"][-1] < -0.3):
                    print "Weird supernova!", snpath

                the_data["mBx1c_cov_list"] = concatenate((the_data["mBx1c_cov_list"], array([[[mBmB, mBx1, mBc],
                                                                                              [mBx1, x1x1, x1c],
                                                                                              [mBc, x1c, cc]]], dtype=float64)   ), axis = 0)

                dparam_dzps = get_dparam_dzps(snpath + "/result_deriv.dat")
                
                for key in dparam_dzps:
                    # key is (Lambda or Zeropoint, Instrument|Band)
                    key_parts = key[1].split("|") #  Instrument, Band

                    if not the_data["efflambs"].has_key(key):
                        the_data["efflambs"][key] = get_efflamb(key_parts[0], key_parts[1])

                    if key[0] == "Zeropoint":
                        zps_from_band = band_to_zps(key_parts[0], key_parts[1], the_data["efflambs"][key])

                        for zp_from_band in zps_from_band:
                            # zp_from_band is band_name, band_d/dzp
                            
                            if not the_data["calib_names"].count(("Zeropoint", zp_from_band[0])):
                                the_data["calib_names"].append(("Zeropoint", zp_from_band[0]))
                            calib_ind = the_data["calib_names"].index(("Zeropoint", zp_from_band[0]))
                            the_data["d_mBx1c_dcalib_list"][current_sn_ind, :, calib_ind] = dparam_dzps[key]*zp_from_band[1]
                    else:
                        if not the_data["calib_names"].count(key):
                            the_data["calib_names"].append(key)
                        calib_ind = the_data["calib_names"].index(key)
                        the_data["d_mBx1c_dcalib_list"][current_sn_ind, :, calib_ind] = dparam_dzps[key]

                current_sn_ind += 1
            else:
                print "Skipping...", snpath,
                for j in range(len(okay_names)):
                    if not okay_to_add[j]:
                        print okay_names[j],
                print

    the_data["calib_unceratainties"] = get_calib_uncertainties(the_data["calib_names"], params["calib_errs"])
    for i in range(len(the_data["calib_names"])):
        print the_data["calib_names"][i], the_data["calib_unceratainties"][i]


    the_data["d_mBx1c_dcalib_list"] = the_data["d_mBx1c_dcalib_list"][:len(the_data["mB_list"]), :, :len(the_data["calib_names"])]
    print 'the_data["d_mBx1c_dcalib_list"].shape ', the_data["d_mBx1c_dcalib_list"].shape
    print 'the_data["calib_names"] ', the_data["calib_names"]
    

    print "read cov shape ", the_data["mBx1c_cov_list"].shape

    for current_sample in range(len(the_data["sample_names"])):
        inds = where(the_data["sample_list"] == current_sample)
        plt.subplot(2,1,1)
        plt.errorbar(the_data["z_CMB_list"][inds], the_data["c_list"][inds], yerr = sqrt(the_data["mBx1c_cov_list"][:,2,2][inds]), fmt ='.', capsize = 0)
        plt.xlim(0., 0.1)
        plt.subplot(2,1,2)
        plt.errorbar(the_data["z_CMB_list"][inds], the_data["c_list"][inds], yerr = sqrt(the_data["mBx1c_cov_list"][:,2,2][inds]), fmt ='.', capsize = 0)

    plt.savefig("c_vs_z.pdf")
    f_read.close()

    f = open(params["mag_cut"])
    lines = f.read().split('\n')
    f.close()

    for filename in filenamelist:
        found = 0
        for line in lines:
            parsed = line.split(None)
            if len(parsed) == 3:
                if parsed[0] == filename.split("/")[-1]:
                    found += 1
                    the_data["mag_cut_list"] = append(the_data["mag_cut_list"], float(parsed[1]))
                    the_data["mag_cut_disp_list"] = append(the_data["mag_cut_disp_list"], float(parsed[2]))
        assert found == 1, "Check mag_cut.txt! " + filename



    return the_data


################################################# Redshifts for Integration ###################################################

def get_redshifts(redshifts):
    appended_redshifts = arange(0., 2.51, 0.1)
    tmp_redshifts = concatenate((redshifts, appended_redshifts))
    
    sort_inds = list(argsort(tmp_redshifts))
    unsort_inds = [sort_inds.index(i) for i in range(len(tmp_redshifts))]
    
    tmp_redshifts = sort(tmp_redshifts)
    redshifts_sort_fill = sort(concatenate((tmp_redshifts, 0.5*(tmp_redshifts[1:] + tmp_redshifts[:-1]))))
    
    return redshifts, redshifts_sort_fill, unsort_inds, len(appended_redshifts)


def get_redshift_coeffs(sample_list, z_list):
    redshift_coeffs = [[], [], [], []]

    for i in range(len(z_list)):
        inds = where(sample_list == sample_list[i])
        sample_z = z_list[inds]

        minz = min(sample_z)
        maxz = max(sample_z)
        assert maxz > minz, "Only one redshift? " + str(sample)

        for j in range(4):
            coeffs = zeros(4.)
            coeffs[j] = 1
            ifn = interp1d([minz, minz*2./3. + maxz/3., minz/3. + maxz*2./3., maxz], coeffs, kind = 'linear')
            redshift_coeffs[j].append(ifn(z_list[i]))

    #plt.figure(2)
    #for i in range(len(z_list)):
    #    for j in range(4):
    #        plt.plot(z_list[i], redshift_coeffs[j][i], '.', color = 'bcgr'[j])
    #plt.savefig("redshift_coeffs.pdf")
    
    return transpose(array(redshift_coeffs))


################################################# Init FN ###################################################

def init_fn():
    n_sne = len(the_data["x1_list"])
    n_samples = len(the_data["sample_names"])
    print "n_sne ", n_sne
    print "n_samples ", n_samples
            
    return {"MB": random.random()*0.2 - 19.1,
            "Om": random.random()*0.4 + 0.1,
            "alpha_angle_low": arctan(random.random()*0.2),
            "alpha_angle_high": arctan(random.random()*0.2),
            "beta_angle_blue": arctan(random.random()*0.5 + 2.5),
            "beta_angle_red": arctan(random.random()*0.5 + 2.5),
            "log10_sigma_int": log10(random.random(size = n_samples)*0.1 + 0.1),
            "mBx1c_int_variance": [0.9, 0.05, 0.05],
            "Lmat": [[1.0, 0.0, 0.0],
                     [random.random()*0.1 - 0.05, random.random()*0.1 + 0.7, 0.0],
                     [random.random()*0.1 - 0.05, random.random()*0.1 - 0.05, random.random()*0.1 + 0.7]],
            #"mass_0": 10,
            "delta_0": random.random()*0.05,
            "delta_h": 0.5,
            "calibs": random.normal(size = len(the_data["calib_names"]))*0.01,
            "blind_values": [0.]*n_samples,
            
            "true_c": random.random(size = n_sne)*0.02 - 0.01 + clip(the_data["c_list"], -0.2, 1.0),
            "true_x1": random.random(size = n_sne)*0.2 - 0.1 + the_data["x1_list"],
            
            "x1_star": random.random(size = [len(the_data["sample_names"]), stan_data["n_x1c_star"]])*0.05,
            "c_star": random.random(size = [len(the_data["sample_names"]), stan_data["n_x1c_star"]])*0.05,
            #"alpha_c": random.random(size = [len(the_data["sample_names"]), stan_data["n_x1c_star"]])*2. - 1.,
            "delta_c": random.random(size = [len(the_data["sample_names"]), stan_data["n_x1c_star"]])*0.2 - 0.1,
            "log10_R_x1": random.random(size = n_samples)*0.5 - 0.25,
            "log10_R_c": random.random(size = n_samples)*0.4 - 1.2,

            "outl_frac": random.random()*0.02 + 0.01,
        }
            
            

################################################# Main Program ###################################################


params = helper_functions.get_params(sys.argv[1])

################################################# And Go! ###################################################
assert params["iter"] % 4 == 0, "iter should be a multiple of four! "  + str(params["iter"])

the_data = read_data(params)
samples_txt = "_".join([item.split(".")[0].split("/")[-1] for item in params["filenamelist"]])

for i, sample in enumerate(the_data["sample_names"]):
    print sample, sum(the_data["sample_list"] == i)


n_sne = len(the_data["c_list"])

obs_mBx1c = []
obs_mBx1c_cov = zeros((3*n_sne, 3*n_sne), dtype=float64)

for i in range(n_sne):
    obs_mBx1c.append([the_data["mB_list"][i], the_data["x1_list"][i], the_data["c_list"][i]])

obs_mBx1c_cov = the_data["mBx1c_cov_list"]

redshifts, redshifts_sort_fill, unsort_inds, nzadd = get_redshifts(the_data["z_CMB_list"])


stan_data = {"n_sne": n_sne, "nzadd": nzadd,
             "n_samples": len(the_data["sample_names"]),
             "redshift_coeffs": get_redshift_coeffs(the_data["sample_list"], the_data["z_CMB_list"]),
             "n_calib": len(the_data["calib_names"]),
             "d_mBx1c_d_calib": the_data["d_mBx1c_dcalib_list"],
             "calib_uncertainties": the_data["calib_unceratainties"],
             "n_x1c_star": 4, # 3 = quadratic in redshift with old approach
             "mass": the_data["mass"],
             "mass_err": the_data["mass_err"],
             # The +1 here is for Stan's indexing, which is from 1 not 0
             "sample_list": the_data["sample_list"] + 1, "redshifts": redshifts, "redshifts_sort_fill": redshifts_sort_fill, "unsort_inds": unsort_inds,
             "obs_mBx1c": obs_mBx1c,
             "obs_mBx1c_cov": obs_mBx1c_cov,
             "do_blind": params["do_blind"],
             "do_twoalphabeta": params["do_twoalphabeta"],

             "outl_mBx1c_uncertainties": [1.]*3,
             "outl_frac_prior_lnmean": log(params["outl_frac"]),
             "outl_frac_prior_lnwidth": 0.5,

             "mB_cuts": the_data["mag_cut_list"],
             "mB_cut_vars": the_data["mag_cut_disp_list"]**2.
         }

pickle.dump((the_data, stan_data, params), open("inputs_" + samples_txt + ".pickle", "wb"))



print "nzadd ", stan_data['nzadd']
# print stan_data['n_sne']
# print stan_data['n_samples']
# print stan_data['sample_list'].shape
# print stan_data['redshift'].shape
# print stan_data['obs_mBx1c']
# print stan_data['obs_mBx1c_cov'].shape

for blind_iter in range(1*stan_data["do_blind"]):
    print "Iter: ", blind_iter

    fit = pystan.stan(file=params["stan_code"], data=stan_data,
                      iter=params["iter"]/2, chains=params["chains"], n_jobs = params["n_jobs"], refresh = 5, init = init_fn)#, pars = ["beta", "dbeta", "alpha", "dalpha", "MB", "Om", "blind_values"])#, sample_file = "/Users/rubind/Dropbox/samples.txt")
    fit_params = fit.extract(permuted = True)
    fit_params = filter_fit_params(fit_params, "MB", params["chains"], params["iter"]/4) # burns the first half of the chain, so iter/4
    
    print "Doing the blinding..."
    
    assert len(fit_params["blind_values"][0]) == stan_data["n_samples"], "Transpose!"
    
    if stan_data["do_blind"]:
        for i in range(stan_data["n_sne"]):
            stan_data["obs_mBx1c"][i][0] += median(fit_params["blind_values"][:, stan_data["sample_list"][i] - 1])
    else:

        for i, sample_name in enumerate(the_data["sample_names"]): # Printing this violates the blind!
            print sample_name, fit_params["blind_values"][:, i]
            print sample_name, median(fit_params["blind_values"][:, i]), std(fit_params["blind_values"][:, i]), 1.4826*median(abs(fit_params["blind_values"][:, i] - median(fit_params["blind_values"][:, i])))

    summarize_parameters(fit_params, thekeys = ["beta", "dbeta", "alpha", "dalpha", "MB", "Om", "delta_h", "delta_0"])

stan_data["do_blind"] = 0

print "Rerunning..."
fit = pystan.stan(file=params["stan_code"], data=stan_data,
                  iter=params["iter"], chains=params["chains"], n_jobs = params["n_jobs"], refresh = 10, init = init_fn, sample_file = params["sample_file"]
                  # pars = ["beta", "dbeta", "alpha", "dalpha", "MB", "Om", "sigma_int", "x1_star", "R_x1", "c_star", "R_c", "calibs"]
                      )#, sample_file = "/Users/rubind/Dropbox/samples.txt")

print "I hope you have a log file:"

try:
    print fit
except:
    print "Couldn't print fit! Something is very wrong!"

fit_params = fit.extract(permuted = True)

try:
    fit_params = filter_fit_params(fit_params, "MB", params["chains"], params["iter"]/2) # burns the first half of the chain, so iter/2
except:
    print "Couldn't filter bad chains! One or more chains may be bad!"

summarize_parameters(fit_params)

pickle.dump(fit_params, open("samples_" + samples_txt + ".pickle", "wb"))

