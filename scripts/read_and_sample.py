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
import gzip
from FileRead import readcol


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
                
                "mobs_cut0": [],
                "mobs_cut1": [],
                "est_mobs_cuts": [],
                "est_mobs_sigmas": [],
                
                "efflambs": {}, # Filter wavelengths
                "d_mBx1c_dcalib_list": zeros([3000,3,500], dtype=float64), # This is an inefficient way to do this, but this is initialized to fixed size, then trimmed later.

                "photoz_inds": [],
                "d_mBx1c_dz_list": [],
                "photo_z0": [],
                "photo_dz": [],
                "n_photoz": 0, # Number of SNe with photo-z's
                "photo_spikez": []
            }


    current_sn_ind = 0
    
    filenamelist = params["filenamelist"]

    f_read = open("sn_input.txt", 'w')
    f_read.write("#SN\tRA\tDEC\tZHEL\tZCMB\tPASS\n") # List of SNe that pass all cuts

    [magcut_input_fls, magcut_k_correction_fls, magcut_est_cuts, magcut_est_sigmas] = readcol(params["mag_cut"], 'aaff')
    magcut_k_correction_fls = [item.replace("$UNITY", os.environ["UNITY"]) for item in magcut_k_correction_fls]
    
    for current_sample, directory in enumerate(filenamelist):
        the_data["sample_names"].append(directory)

        f = open(directory)
        snpaths = f.read().split('\n')
        snpaths = [item.strip() for item in snpaths]
        snpaths = [item for item in snpaths if item != ""]

        f.close()

        print "current_sample, directory", current_sample, directory
        magcut_ind = magcut_input_fls.index(directory.split("/")[-1])
        print "magcut_ind ", magcut_ind

        the_data["est_mobs_cuts"].append(magcut_est_cuts[magcut_ind])
        the_data["est_mobs_sigmas"].append(magcut_est_sigmas[magcut_ind])
        
        kc_ifn0, kc_ifn1 = helper_functions.get_kcorrect_ifns(magcut_k_correction_fls[magcut_ind])

        for snpath in snpaths:
            this_redshift = helper_functions.read_param(snpath + "/result_salt2.dat", "Redshift")
            this_redshift_cmb = helper_functions.read_param(snpath + "/lightfile", "z_cmb")
            if this_redshift_cmb == None:
                this_redshift_cmb = helper_functions.read_param(snpath + "/lightfile", "z_CMB")

            this_redshift_helio = helper_functions.read_param(snpath + "/lightfile", "z_heliocentric")
            if this_redshift_cmb == None and this_redshift > 0.1:
                this_redshift_cmb = this_redshift
                print "Couldn't find redshift for ", snpath
            if this_redshift_helio == None and this_redshift > 0.1:
                this_redshift_helio = this_redshift
                print "Couldn't find redshift for ", snpath
                
            this_RA = helper_functions.read_param(snpath + "/lightfile", "RA")
            this_DEC = helper_functions.read_param(snpath + "/lightfile", "DEC")
            this_firstphase = helper_functions.read_param(snpath + "/result_salt2.dat", "FirstPhase")
            this_lastphase = helper_functions.read_param(snpath + "/result_salt2.dat", "LastPhase")
            this_colorerr = helper_functions.read_param(snpath + "/result_salt2.dat", "Color", ind = 2)
            this_x1 = helper_functions.read_param(snpath + "/result_salt2.dat", "X1", ind = 1)
            this_x1_err = helper_functions.read_param(snpath + "/result_salt2.dat", "X1", ind = 2)
            try:
                this_check = helper_functions.read_param(snpath + "/result_deriv.dat", "Check", ind = 4)
            except:
                this_check = 1000

            if this_x1_err == None:
                this_x1 = 100.
                this_x1_err = 100.

            weird_sn = helper_functions.read_param(params["weird_sn_list"], snpath.split("/")[-1])
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

                if helper_functions.read_param(snpath + "/lightfile", "Photoz") != None:
                    print "Photoz found!", snpath
                    the_data["n_photoz"] += 1
                    
                    the_data["photoz_inds"].append(the_data["n_photoz"]) # That's right, after incrementing the counter
                    the_data["d_mBx1c_dz_list"].append([helper_functions.read_param(snpath + "/result_deriv.dat", "Redshift", ind = 5),
                                                        helper_functions.read_param(snpath + "/result_deriv.dat", "Redshift", ind = 6),
                                                        helper_functions.read_param(snpath + "/result_deriv.dat", "Redshift", ind = 7)])

                    the_data["photo_z0"].append(helper_functions.read_param(snpath + "/lightfile", "Photoz", ind = 1))
                    the_data["photo_dz"].append(helper_functions.read_param(snpath + "/lightfile", "Photoz", ind = 2))
                    the_data["photo_spikez"].append(this_redshift_helio)
                    
                else:
                    the_data["photoz_inds"].append(0)


                the_data["z_CMB_list"] = append(the_data["z_CMB_list"], this_redshift_cmb
                                                )

                the_data["mobs_cut0"].append(kc_ifn0(this_redshift_helio))
                the_data["mobs_cut1"].append(kc_ifn1(this_redshift_helio))
                

                the_data["mB_list"] = append(the_data["mB_list"],
                                             helper_functions.read_param(snpath + "/result_salt2.dat", "RestFrameMag_0_B"))
                the_data["c_list"] = append(the_data["c_list"],
                                            helper_functions.read_param(snpath + "/result_salt2.dat", "Color"))
                the_data["x1_list"] = append(the_data["x1_list"],
                                             helper_functions.read_param(snpath + "/result_salt2.dat", "X1"))


                the_data["sample_list"] = append(the_data["sample_list"], current_sample)

                this_mass = helper_functions.read_param(snpath + "/lightfile", "Mass", ind = 1)
                this_mass_err = sqrt(abs(
                    helper_functions.read_param(snpath + "/lightfile", "Mass", ind = 2)*helper_functions.read_param(snpath + "/lightfile", "Mass", ind = 3)
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
                mBmB = helper_functions.read_param(snpath + "/result_salt2.dat", "RestFrameMag_0_B", ind = 2)**2. + (params["pec_vel_disp"]/the_data["z_CMB_list"][-1]*5./log(10.))**2. + (params["lensing_disp"]*the_data["z_CMB_list"][-1])**2.
                mBx1 = helper_functions.read_param(snpath + "/result_salt2.dat", "CovRestFrameMag_0_BX1")
                mBc = helper_functions.read_param(snpath + "/result_salt2.dat", "CovColorRestFrameMag_0_B")
                x1x1 = helper_functions.read_param(snpath + "/result_salt2.dat", "CovX1X1")
                x1c = helper_functions.read_param(snpath + "/result_salt2.dat", "CovColorX1")
                cc = helper_functions.read_param(snpath + "/result_salt2.dat", "CovColorColor")

                h_resid = (the_data["mB_list"][-1] - - 19.1 + 0.13*the_data["x1_list"][-1] - 3.*the_data["c_list"][-1]) - (5*log10(the_data["z_CMB_list"][-1]*(1. + the_data["z_CMB_list"][-1])) + 42.9)
                if abs(h_resid) > 2 or (the_data["c_list"][-1] > 1) or (the_data["c_list"][-1] < -0.3):
                    print "Weird supernova!", snpath

                the_data["mBx1c_cov_list"] = concatenate((the_data["mBx1c_cov_list"], array([[[mBmB, mBx1, mBc],
                                                                                              [mBx1, x1x1, x1c],
                                                                                              [mBc, x1c, cc]]], dtype=float64)   ), axis = 0)

                dparam_dzps = helper_functions.get_dparam_dzps(snpath + "/result_deriv.dat")

                for key in dparam_dzps:
                    # key is (Lambda or Zeropoint, Instrument|Band)
                    key_parts = key[1].split("|") #  Instrument, Band

                    
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

    the_data["calib_uncertainties"] = [0.01]*len(the_data["calib_names"])
    for i in range(len(the_data["calib_names"])):
        print the_data["calib_names"][i], the_data["calib_uncertainties"][i]

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

    if the_data["d_mBx1c_dz_list"] == []:
        the_data["d_mBx1c_dz_list"] = zeros([0,3], dtype=float64)

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


def get_redshift_coeffs(sample_list, z_list, n_x1c_star):
    redshift_coeffs = [[] for i in range(n_x1c_star)]

    for i in range(len(z_list)):
        inds = where(sample_list == sample_list[i])
        sample_z = z_list[inds]

        if n_x1c_star > 1:
            minz = min(sample_z)*0.99999
            maxz = max(sample_z)*1.00001
            assert maxz > minz, "Only one redshift? " + str(sample)

            for j in range(n_x1c_star):
                coeffs = zeros(n_x1c_star, dtype=float64)
                coeffs[j] = 1
                ifn = interp1d(linspace(minz, maxz, n_x1c_star), coeffs, kind = 'linear')
                redshift_coeffs[j].append(ifn(z_list[i]))
        else:
            redshift_coeffs[0].append(1.)
    #plt.figure(2)
    #for i in range(len(z_list)):
    #    for j in range(4):
    #        plt.plot(z_list[i], redshift_coeffs[j][i], '.', color = 'bcgr'[j])
    #plt.savefig("redshift_coeffs.pdf")
    
    return transpose(array(redshift_coeffs))

def zcount(z, zmin, zmax):
    return sum((array(z) >= zmin)*(array(z) < zmax))

def add_zbins(stan_data, cosmo_model):
    # For inned mu
    stan_data["zbins"] = [0.99999*stan_data["redshifts"].min()]

    while max(stan_data["zbins"]) < max(stan_data["redshifts"]):
        zstep = 0.125
        while (zcount(stan_data["redshifts"], stan_data["zbins"][-1], stan_data["zbins"][-1]*exp(zstep)) < 10.) and (stan_data["zbins"][-1]*exp(zstep) < stan_data["redshifts"].max()):
            zstep *= 1.5

        stan_data["zbins"].append(stan_data["zbins"][-1]*exp(zstep))



    stan_data["n_zbins"] = len(stan_data["zbins"])

    f = open("zbins.txt", 'w')
    for zbin in stan_data["zbins"]:
        f.write(str(zbin) + '\n')
    f.close()

    plt.figure()
    plt.hist(stan_data["redshifts"])
    plt.plot(stan_data["zbins"], [100]*stan_data["n_zbins"], '.', color = 'k')
    plt.savefig("redshift_binning.pdf")
    plt.close()

    stan_data["dmu_dbin"] = zeros([stan_data["n_sne"], stan_data["n_zbins"]], dtype=float64)
    stan_data["dmudz_dbin"] = zeros([stan_data["n_sne"], stan_data["n_zbins"]], dtype=float64)

    for j in range(stan_data["n_zbins"]):
        nodes = zeros(stan_data["n_zbins"], dtype=float64)
        nodes[j] = 1.

        ifn = interp1d(stan_data["zbins"], nodes, kind = 'quadratic')
        for i in range(stan_data["n_sne"]):
            stan_data["dmu_dbin"][i, j] = ifn(stan_data["redshifts"][i])
            stan_data["dmudz_dbin"][i, j] = (ifn(stan_data["redshifts"][i] + 0.001) - ifn(stan_data["redshifts"][i]))/0.001


    plt.figure()
    plt.imshow(stan_data["dmu_dbin"])
    plt.savefig("dmu_dbin.pdf")
    plt.close()
        
    stan_data["cosmo_model"] = cosmo_model

    return stan_data


################################################# Init FN ###################################################

def init_fn():
    n_sne = len(the_data["x1_list"])
    n_samples = len(the_data["sample_names"])
    print "n_sne ", n_sne
    print "n_samples ", n_samples
            
    return {"MB": random.random()*0.2 - 19.1,
            "Om": random.random()*0.4 + 0.1,
            "alpha_angle": arctan(random.random()*0.2),
            "beta_angle_blue": arctan(random.random()*0.5 + 2.5),
            "beta_angle_red": arctan(random.random()*0.5 + 2.5),
            "log10_sigma_int": log10(random.random(size = n_samples)*0.1 + 0.1),
            "mBx1c_int_variance": [0.9, 0.05, 0.05],
            #"mass_0": 10,
            "delta_0": random.random()*0.05,
            "delta_h": 0.5,
            "calibs": random.normal(size = len(the_data["calib_names"]))*0.01,
            "blind_values": [0.]*n_samples,
            
            "true_cB": random.random(size = n_sne)*0.02 - 0.01 + clip(the_data["c_list"]/2., -0.2, 1.0),
            "true_cR": random.random(size = n_sne)*0.01 + clip(the_data["c_list"]/2., 0, 1.0),
            "true_x1": random.random(size = n_sne)*0.2 - 0.1 + the_data["x1_list"],
            
            "x1_star": random.random(size = [len(the_data["sample_names"]), stan_data["n_x1c_star"]])*0.05,
            "c_star": random.random(size = [len(the_data["sample_names"]), stan_data["n_x1c_star"]])*0.05,
            "log10_R_x1": random.random(size = n_samples)*0.5 - 0.25,
            "log10_R_c": random.random(size = [len(the_data["sample_names"]), stan_data["n_x1c_star"]])*0.4 - 1.2,
            "log10_tau_c": random.random(size = [len(the_data["sample_names"]), stan_data["n_x1c_star"]])*0.4 - 1.2,

            "outl_frac": random.random()*0.02 + 0.01,
            "mobs_cuts": stan_data["est_mobs_cuts"] + random.normal(size = n_samples)*0.1, "mobs_cut_sigmas": [0.5]*n_samples,

            "dz": random.normal(size = stan_data["n_photoz"])*0.01
        }
            
            

################################################# Main Program ###################################################

inputfl = sys.argv[1]
cosmo_model = int(sys.argv[2])


if inputfl.count("pickle"):
    (the_data, stan_data, params) = pickle.load(gzip.open(inputfl, "rb"))
else:
    params = helper_functions.get_params(inputfl)
    n_x1c_star = 2

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
                 "redshift_coeffs": get_redshift_coeffs(the_data["sample_list"], the_data["z_CMB_list"], n_x1c_star),
                 "n_calib": len(the_data["calib_names"]),
                 "d_mBx1c_d_calib": the_data["d_mBx1c_dcalib_list"],
                 "calib_uncertainties": the_data["calib_uncertainties"],
                 "n_x1c_star": n_x1c_star, # 3 = quadratic in redshift with old approach
                 "mass": the_data["mass"],
                 "mass_err": the_data["mass_err"],
                 # The +1 here is for Stan's indexing, which is from 1 not 0
                 "sample_list": the_data["sample_list"] + 1, "redshifts": redshifts, "redshifts_sort_fill": redshifts_sort_fill, "unsort_inds": unsort_inds,
                 "obs_mBx1c": array(obs_mBx1c),
                 "obs_mBx1c_cov": array(obs_mBx1c_cov),
                 "do_blind": params["do_blind"],
                 "do_twoalphabeta": params["do_twoalphabeta"],

                 "outl_frac_prior_lnmean": log(params["outl_frac"]),
                 "outl_frac_prior_lnwidth": 0.5,

                 "n_photoz": the_data["n_photoz"],
                 "d_mBx1c_dz_list": the_data["d_mBx1c_dz_list"],
                 "photo_z0": the_data["photo_z0"],
                 "photo_dz": the_data["photo_dz"],
                 "spike_redshift_prob": [0.8]*the_data["n_photoz"],
                 "photoz_inds": the_data["photoz_inds"],
                 "photo_spikez": the_data["photo_spikez"],


                 "est_mobs_cuts": the_data["est_mobs_cuts"],
                 "est_mobs_sigmas": the_data["est_mobs_sigmas"],
                 "mobs_cut0": the_data["mobs_cut0"], "mobs_cut1": the_data["mobs_cut1"]
             }

    plt.figure()
    plt.plot(stan_data["redshifts"], stan_data["mobs_cut0"], '.')
    plt.plot(stan_data["redshifts"], stan_data["mobs_cut1"], '.')
    plt.savefig("k_corr_check.pdf")
    plt.close()

    pickle.dump((the_data, stan_data, params), gzip.open("inputs_" + samples_txt + ".pickle", "wb"))


stan_data = add_zbins(stan_data, cosmo_model)

print "nzadd ", stan_data['nzadd']
# print stan_data['n_sne']
# print stan_data['n_samples']
# print stan_data['sample_list'].shape
# print stan_data['redshift'].shape
# print stan_data['obs_mBx1c']
# print stan_data['obs_mBx1c_cov'].shape

if stan_data["do_blind"]:
    print "Blinding!"
    # There are two phases of blinding:
    # -Making the best-fit Om = 0.3
    # -Bringing all samples into alignment with -19.1 given Om = 0.3 (eventually!)

    [zblind, mublind, dmublinddOm] = readcol(os.environ["UNITY"] + "/paramfiles/z_mu_dmudOm.txt", 'fff')
    mublindfn = interp1d(zblind, mublind, kind = 'linear')
    dmublinddOmfn = interp1d(zblind, dmublinddOm, kind = 'linear')

    for i in range(2):
        muobs = stan_data["obs_mBx1c"][:,0] + 0.14*stan_data["obs_mBx1c"][:,1] - 3.1*stan_data["obs_mBx1c"][:,2] - -  19.1
        dmuobs = sqrt(0.15**2. + stan_data["obs_mBx1c_cov"][:,0,0] + 0.14**2. * stan_data["obs_mBx1c_cov"][:, 1,1] + 3.1**2. * stan_data["obs_mBx1c_cov"][:, 2,2]) # Doesn't have to be exact

        jmat = zeros([stan_data["n_sne"], 2], dtype=float64)
        jmat[:,0] = 1.
        jmat[:,1] = dmublinddOmfn(stan_data["redshifts"])
        wmat = diag(dmuobs**-2.)

        bestvals = dot(
            linalg.inv(dot(transpose(jmat), dot(wmat, jmat))),
            dot(transpose(jmat), dot(wmat, muobs - mublindfn(stan_data["redshifts"])))
            )

        stan_data["obs_mBx1c"][:,0] -= bestvals[0] + bestvals[1]*dmublinddOmfn(stan_data["redshifts"])
        the_data["mB_list"] = array(the_data["mB_list"]) - bestvals[0] - bestvals[1]*dmublinddOmfn(stan_data["redshifts"])

        if i > 0:
            print bestvals
            assert all(abs(bestvals) < 1e-3)
            print "Blinding passed!"

        
        

print "Running..."
fit = pystan.stan(file=params["stan_code"], data=stan_data,
                  iter=params["iter"], chains=params["chains"], n_jobs = params["n_jobs"], refresh = 10, init = init_fn, sample_file = params["sample_file"]
                  # pars = ["beta", "dbeta", "alpha", "dalpha", "MB", "Om", "sigma_int", "x1_star", "R_x1", "c_star", "R_c", "calibs"]
                      )#, sample_file = "/Users/rubind/Dropbox/samples.txt")


fit_params = fit.extract(permuted = True)

try:
    fit_params = filter_fit_params(fit_params, "MB", params["chains"], params["iter"]/2) # burns the first half of the chain, so iter/2
except:
    print "Couldn't filter bad chains! One or more chains may be bad!"

#summarize_parameters(fit_params)


try:
    samples_txt
    pickle.dump(fit_params, gzip.open("samples_" + samples_txt + ".pickle", "wb"))
except:
    pickle.dump(fit_params, gzip.open("samples.pickle", "wb"))


print "I hope you have a log file:"

try:
    print fit
except:
    print "Couldn't print fit! Something is very wrong!"
