from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
import numpy as np
from FileRead import readcol, file_to_fn
from scipy.interpolate import interp1d
import sncosmo
from astropy.table import Table
import subprocess
import tqdm
import pickle
import os
import sys
from astropy.cosmology import FlatLambdaCDM
import argparse
from scipy.special import erf
import Spectra


true_H0 = 71.

def do_it(cmd):
    print(cmd)
    return subprocess.getoutput(cmd)
    

def band_to_instr(band):
    if band[:4] == "sdss":
        return "SDSS"
    elif band[:4] == "uvot":
        return "UVOT"
    elif band[:4] == "Bess":
        return "OtherNearby"
    elif band[:2] == "f1":
        return "WFC3"
    else:
        return "ACSWF"

def band_to_name(band):
    if band[:4] == "sdss":
        return "SDSS_" + band[-1]
    elif band[:4] == "uvot":
        return "UVOT_" + band[-1].lower()
    elif band[:2] == "f1":
        return "WFC3_" + band.lower()
    elif band.count("Bessell") == 1:
        return band
    else:
        return band.upper()
    

def get_SNCosmo_model(these_params, source):
    #sncosmo_model = sncosmo.Model(source="salt2-extended")
    sncosmo_model = sncosmo.Model(source=source)

    tmp_params = dict(these_params)
    for nonSALTkey in nonSALTkeys:
        del tmp_params[nonSALTkey]
    sncosmo_model.set(**tmp_params)

    
    ten_pc_z = 2.33494867e-9 * (true_H0/70.)
    cosmo = FlatLambdaCDM(Om0 = 0.3, H0 = true_H0)
    assert abs(cosmo.distmod(z=ten_pc_z).value) < 1.e-3, "Distance modulus zeropoint wrong!"

    ampl = 1.
    for i in range(2):
        sncosmo_model.set(z=ten_pc_z, t0=0., x0=ampl)

        mag = sncosmo_model.bandmag('bessellb', 'ab', 0.)
        #print "mag, ampl ", mag, ampl
        ampl *= 10.**(0.4*(mag - these_params["MB"]))
        
    #print "mu ", mu

    sncosmo_model.set(z=these_params["z"], t0=these_params["t0"], x0 = ampl*10**(-0.4*these_params["latentmu"]))
    return sncosmo_model


def approxmB(model, date, redshift):
    """Only used for rest-frame mag selection, rather than observer-frame"""
    x0 = model.get("x0")
    t0 = model.get("t0")
    width_around_t0 = 10*(1. + redshift)

    return -2.5*np.log10(x0 * np.exp(  -0.5*(   (date - t0)/width_around_t0   )**2.  )
                         )

def mag_to_negflux(mag, obs_err):
    return -10.**(-0.4*(mag - 27.5)) + np.random.normal()*obs_err


def get_observed_SNe_followup_limited(nsne, dates, all_SNe, model, z_range_key, obs_err):
    observed_SNe = np.array([all_SNe[i]["z"] < 0.01 for i in range(nsne)], dtype=np.int32) #np.zeros(nsne, dtype=np.int16)
    
    for night in dates:
        all_mags = []
        
        for i in range(nsne):
            if observed_SNe[i] == 0 and np.abs(night - all_SNe[i]["t0"]) < 50:
                model = get_SNCosmo_model(all_SNe[i], source)
                if params["obs_mag_selection"]:
                    if z_range_key == "L" or z_range_key == "S":
                        all_mags.append(mag_to_negflux(model.bandmag("sdssr", "ab", night), obs_err = obs_err))
                    elif z_range_key == "H":
                        all_mags.append(mag_to_negflux(model.bandmag("sdssi", "ab", night), obs_err = obs_err))
                    else:
                        raise Exception("Unknown z_range_key " + z_range_key)
                else:
                    all_mags.append(mag_to_negflux(approxmB(model, night, all_SNe[i]["z"]), obs_err = obs_err))
            else:
                all_mags.append(1e20)
                
        all_mags = np.array(all_mags)
        print("all_mags", all_mags)
                                
        inds = np.argsort(all_mags)

        for i in range(params["nsnepernight"]):
            observed_SNe[inds[i]] = 1
    return observed_SNe


def get_observed_SNe_mag_limited(nsne, dates, all_SNe, model, z_range_key, mag_limit, sigma_mag_limit):
    observed_SNe = np.zeros(nsne, dtype=np.int16)
    
    for night in tqdm.tqdm(dates):
        all_mags = []
        
        for i in range(nsne):
            if observed_SNe[i] == 0 and np.abs(night - all_SNe[i]["t0"]) < 50:
                model = get_SNCosmo_model(all_SNe[i], source)
                if params["obs_mag_selection"]:
                    if z_range_key == "V":
                        try:
                            all_mags.append(model.bandmag("f125w", "ab", night))
                        except:
                            all_mags.append(10.)
                    else:
                        raise Exception("Unknown z_range_key " + z_range_key)
                else:
                    all_mags.append(approxmB(model, night, all_SNe[i]["z"]))
            else:
                all_mags.append(1e20)
                
        all_mags = np.array(all_mags)
        print("all_mags", all_mags)

        for i in range(nsne):
            if all_mags[i] + np.random.normal()*sigma_mag_limit < mag_limit:
                observed_SNe[i] = 1
                
    return observed_SNe



def get_observed_SNe_volume_limited(nsne, dates, all_SNe, model):
    observed_SNe = np.zeros(nsne, dtype=np.int16)
    
    for night in dates:
        all_mags = []
        
        for i in range(nsne):
            if observed_SNe[i] == 0 and np.abs(night - all_SNe[i]["t0"]) < 50:
                all_mags.append(5*np.log10(all_SNe[i]["z"]))
            else:
                all_mags.append(1e20)

        all_mags = np.array(all_mags)
        print("all_mags", all_mags)
        
        inds = np.argsort(all_mags)

        for i in range(params["nsnepernight"]):
            observed_SNe[inds[i]] = 1
    return observed_SNe


def generate_p_for_one_SN(params, z, min_date, max_date):
    if np.random.random() < params["outlierfrac"]:
        # Outlier
        p = dict(z = z, t0 = np.random.uniform(min_date, max_date),
                 outlier = 1,
                 MB = params["MB"] + np.random.normal()*0.5,
                 x1 = np.random.normal()*2,
                 c = np.random.normal()*0.2,
                 mass = 10. + np.random.normal(),

                 latentMB = np.sqrt(-1.),
                 latentx1 = np.sqrt(-1.),
                 latentc = np.sqrt(-1.),
                 latentcB = np.sqrt(-1.),
                 latentcR = np.sqrt(-1.),
                 beta_R = np.sqrt(-1.),
                 delta_mBx1c = np.sqrt(-1*np.ones(3)),
                 delta_mB = np.sqrt(-1.),
                 delta_x1 = np.sqrt(-1.),
                 delta_c = np.sqrt(-1.),
                 delta_mu = np.sqrt(-1.),
                 x1_slow = np.sqrt(-1.))

    else:
        p = dict(z = z,
                 outlier = 0,
                 t0 = np.random.uniform(min_date, max_date))
        
        if z >= 0.01:
            pec_vel_variance = (0.00217/z)**2.
            p["mass"] = 10. + np.random.normal()
        else:
            # Calibrator
            pec_vel_variance = 0.
            p["mass"] = 10.4 + np.random.normal()

        relative_step_z = 1.9/(1. + 0.9*10.**(0.95*p["z"]))
        relative_step_z = relative_step_z*(1 - params["delta_h"]) + params["delta_h"]
        P_high_eff = relative_step_z * 0.5*(1. + erf(   (p["mass"] - 10.)/(1.414*np.sqrt(0.03**2. + params["step_width"]**2.))   ))

        mass_term = -params["delta"]*P_high_eff

            
        if opts.simtype == "Union3":
            p.update(latentx1 = np.random.normal()*params["Rx1"] + (np.random.exponential() - 1.)*params["tau_x1"],
                     latentcB = np.random.normal()*params["Rc"],
                     beta_R = params["beta_R"] + np.random.normal()*params["sigma_beta_R"],
                     latentcR = (np.random.exponential() - 1.)*params["tau_c"])
        else:
            p["x1_slow"] = np.random.random() <= params["frac_x1_slow_" + "high"*(p["mass"] > 10) + "low"*(p["mass"] <= 10)]
            slow_fast = ["fast", "slow"][p["x1_slow"]]

            p.update(latentx1 = np.random.normal()*params["Rx1_" + slow_fast] + params["x1_star_" + slow_fast],
                     latentcB = np.random.normal()*params["Rc_" + slow_fast] + params["c_star_" + slow_fast],
                     beta_R = params["beta_R"] + np.random.normal()*params["sigma_beta_R"],
                     latentcR = (np.random.exponential() - 1.)*params["tau_c"])

            
        p["latentc"] = p["latentcB"] + p["latentcR"]
        p["beta_R"] += relative_step_z*(0.5 - P_high_eff)*params["delta_beta_R"]

        p["latentMB"] = params["MB"] + (1. - p["x1_slow"])*params["MB_fast_minus_slow"] - params["alpha"]*p["latentx1"] + params["beta_B"]*p["latentcB"] + p["beta_R"]*p["latentcR"] + mass_term


        if opts.simtype == "Union3":
            p["delta_mBx1c"] = np.random.normal(size = 3)*np.array([np.sqrt(params["sigma_unexplained_3d"][0]**2. + (0.055*z)**2. + pec_vel_variance),
                                                                    params["sigma_unexplained_3d"][1],
                                                                    params["sigma_unexplained_3d"][2]])
        else:
            p["delta_mBx1c"] = np.random.normal(size = 3)*np.array([np.sqrt(params["sigma_unexplained_3d_" + slow_fast][0]**2. + (0.055*z)**2. + pec_vel_variance),
                                                                    params["sigma_unexplained_3d_" + slow_fast][1],
                                                                    params["sigma_unexplained_3d_" + slow_fast][2]])
 
            
        p["MB"] = p["latentMB"] + p["delta_mBx1c"][0]
        p["x1"] = p["latentx1"] + p["delta_mBx1c"][1]
        p["c"] = p["latentc"] + p["delta_mBx1c"][2]


        p["delta_mB"] = p["delta_mBx1c"][0]
        p["delta_x1"] = p["delta_mBx1c"][1]
        p["delta_c"] = p["delta_mBx1c"][2]

        p["delta_mu"] = p["delta_mB"] + 0.14*p["delta_x1"] - 3.*p["delta_c"]
        
    cosmo = FlatLambdaCDM(Om0 = 0.3, H0 = true_H0)
    p["latentmu"] = cosmo.distmod(p["z"]).value
    return p

    
def make_dataset(wd, cal_offsets, dataset_ind):
    z_range_key = "ABC"

    for letter_to_look_for in "SLHV":
        if wd.count("_" + letter_to_look_for + "_"):
            assert z_range_key == "ABC"
            z_range_key = letter_to_look_for

            
    if z_range_key == "L":
        # Swope g
        #20, 50, 80 [20.30159178 21.16237294 21.76663019]
        #20, 50, 80 visits [10. 16. 26.]
        #20, 50, 80 cadences [2.    3.    3.998]

        # Foundation r
        #20, 50, 80 [20.00699841 20.42337682 20.8160337 ]
        #20, 50, 80 visits [ 5.  6. 10.]
        #20, 50, 80 cadences [5.0454 7.098  9.498 ]

        
        obs_err = 200. # ZP = 27.5, so depth of 20.0 at 5 sigma
        bands_to_use = ['sdssg', 'sdssr', 'sdssi', 'sdssz']
    elif z_range_key == "S":
        # Union3/SOUSA/iPTF14bdn/lc2fit_UVOT_u.dat 17.206704149882405
        #20, 50, 80 [16.65705729 17.46923784 18.15808045]
        #20, 50, 80 visits [ 6.  11.  16.2]
        #20, 50, 80 cadences [1.97422  2.305875 3.24079 ]

        #Union3/SOUSA/iPTF14bdn/lc2fit_UVOT_v.dat 17.23308259619051
        #20, 50, 80 [15.92489697 17.0413714  17.59495857]
        #20, 50, 80 visits [ 6. 11. 16.]
        #20, 50, 80 cadences [1.97078  2.315325 3.30132 ]

        # lc2fit_LOSS_N2_B.dat 
        # 20, 50, 80 [18.73994798 19.51205921 20.27676355]
        # 20, 50, 80 visits [ 5. 12. 23.]
        # 20, 50, 80 cadences [1.97 2.98 6.96]

        # lc2fit_LOSS_N2_V.dat 
        # 20, 50, 80 [18.52533344 19.36517593 19.97763132]
        # 20, 50, 80 visits [ 5. 14. 24.]
        # 20, 50, 80 cadences [1.98  2.99  6.006]
        
        # lc2fit_LOSS_N2_I.dat 
        # 20, 50, 80 [17.6603694  18.64230244 19.21850216]
        # 20, 50, 80 visits [ 5.  13.  23.8]
        # 20, 50, 80 cadences [1.99  2.99  5.972]

        
        obs_err = 500. # ZP = 27.5, so 5 sigma depth of 19
        bands_to_use = ["Bessell12_B", "Bessell12_V", "Bessell12_R", "Bessell12_I"] #['uvot::u', 'uvot::b', 'uvot::v']
    elif z_range_key == "H":
        obs_err = 5. # Depth of 24.0 at 5 sigma
        bands_to_use = ['sdssg', 'sdssr', 'sdssi', 'sdssz']
    elif z_range_key == "V":
        # For MCT, median F160W depth is 26.146, F125W is 26.442, F850LP is 26.164, F775W is 25.718
        # For MCT, median F160W visits are 6.5, F125W are 6, F850LP 1, and F775W 3.5
        # For MCT, median F160W cadences are 18.2, F125W are 18.7, F850LP 16.3, and F775W is 17.15
        obs_err = 0.7 # Depth of 26.14 at 5 sigma
        bands_to_use = ['f775w', 'f850lp', 'f125w', 'f160w']
    else:
        raise Exception("Unknown z_range_key " + z_range_key)


    model = sncosmo.Model(source=source)

    if z_range_key == "L" or z_range_key == "S":
        if z_range_key == "L":
            zmax = 0.1
        else:
            zmax = 0.05
            
        dates = np.arange(int(np.around(opts.nnearbyperset/3.)), dtype=np.float64)*params["cadence"]

        zlist = []
        while len(zlist) < opts.nnearbyperset*8: # Go 1.5 mag deeper than the selection limit
            zlist += list(sncosmo.zdist(0.01, zmax = zmax, time=dates[-1] - dates[0] - 4*params["cadence"], area=1000., ratefunc = SN_rate_function))
        min_date = dates[0] + params["cadence"]*2
        max_date = dates[-1] - params["cadence"]*2


        calibrator_count = sum([(item < 0.01) for item in zlist])
        print("calibrator_count from just volume", calibrator_count)
        zlist = np.concatenate((zlist, 0.008*np.random.random(size = opts.ncalibperset - calibrator_count)))

        
    elif z_range_key == "H":
        dates = np.arange(params["n_visit"], dtype=np.float64)*params["cadence"]
        zlist = [0]
        while np.min(zlist) < 0.01:
            zlist = list(sncosmo.zdist(0., zmax = 1.0, time=dates[-1] - dates[0] - 4*params["cadence"], area=params["ndeg2"], ratefunc = SN_rate_function))
        min_date = dates[0] + params["cadence"]*2
        max_date = dates[-1] - params["cadence"]*2
    elif z_range_key == "V":
        dates = np.arange(params["HST_visit"], dtype=np.float64)*params["HST_cadence"]
        min_date = dates[0] + params["HST_cadence"]
        max_date = dates[-1] - params["HST_cadence"]*3
        zlist = [0]
        while np.min(zlist) < 0.01:
            zlist = list(sncosmo.zdist(0., zmax = 3.2, time=dates[-1] - dates[0] - 4*params["HST_cadence"], area=4.0, ratefunc = SN_rate_function))
    else:
        raise Exception("Unknown z_range_key " + z_range_key)

    print("len(zlist)", len(zlist))

    nsne = len(zlist)
    
    all_SNe = []
    for z in zlist:        
        p = generate_p_for_one_SN(params = params, z = z, min_date = min_date, max_date = max_date)
        all_SNe.append(p)



    if params["volume_limited"] == 0:
        if z_range_key == "V":
            if params["obs_mag_selection"]:
                observed_SNe = get_observed_SNe_mag_limited(nsne = nsne, dates = dates, all_SNe = all_SNe, model = model, mag_limit = 26.0, sigma_mag_limit = 0.25, z_range_key = z_range_key)
            else:
                observed_SNe = get_observed_SNe_mag_limited(nsne = nsne, dates = dates, all_SNe = all_SNe, model = model, mag_limit = 16.5, sigma_mag_limit = 0.25, z_range_key = z_range_key)
        else:
            observed_SNe = get_observed_SNe_followup_limited(nsne = nsne, dates = dates, all_SNe = all_SNe, model = model, z_range_key = z_range_key, obs_err = obs_err)
    else:
        observed_SNe = get_observed_SNe_volume_limited(nsne = nsne, dates = dates, all_SNe = all_SNe, model = model)


    NA, bins, NA = plt.hist([all_SNe[i]["z"] for i in range(nsne)], bins = 60)
    plt.close()
    
    plt.hist([all_SNe[i]["z"] for i in range(nsne) if observed_SNe[i] == 1], alpha = 0.5, label = str(sum(observed_SNe)), bins = bins)
    plt.hist([all_SNe[i]["z"] for i in range(nsne) if observed_SNe[i] == 0], alpha = 0.5, label = str(nsne - sum(observed_SNe)), bins = bins)
    plt.legend(loc = 'best')
    plt.savefig("selected_%s.pdf" % z_range_key)
    plt.close()

    peak_mags = []
    SNe_x0s = []
    
    for i in range(nsne):
        model = get_SNCosmo_model(all_SNe[i], source)
        if params["obs_mag_selection"]:
            if z_range_key == "L" or z_range_key == "S":
                peak_mags.append(model.bandmag("sdssr", "ab", all_SNe[i]["t0"]))
            elif z_range_key == "H":
                peak_mags.append(model.bandmag("sdssi", "ab", all_SNe[i]["t0"]))
            elif z_range_key == "V":
                try:
                    peak_mags.append(model.bandmag("f125w", "ab", all_SNe[i]["t0"]))
                except:
                    assert all_SNe[i]["z"] < 0.5
                    peak_mags.append(model.bandmag("f850lp", "ab", all_SNe[i]["t0"]))

        else:
            peak_mags.append(approxmB(model, all_SNe[i]["t0"], all_SNe[i]["z"]))
        SNe_x0s.append(model.get("x0"))
        
    peak_mags = np.array(peak_mags)

    NA, bins, NA = plt.hist(peak_mags, bins = 30)
    plt.close()
    
    plt.hist(peak_mags[np.where(observed_SNe)], alpha = 0.5, label = str(sum(observed_SNe)), bins = bins)
    plt.hist(peak_mags[np.where(1 - observed_SNe)], alpha = 0.5, label = str(nsne - sum(observed_SNe)), bins = bins)
    plt.legend(loc = 'best')
    plt.savefig("selected_mags_%s.pdf" % z_range_key)
    plt.close()
    

    print("total observed_SNe", sum(observed_SNe))


    p_wd = wd.replace("dataset_", "UNITY_") + "/SN_params/"
    do_it("mkdir -p " + p_wd)


    f_ladder = open(opts.prefixname + "/distance_ladder_true_vals_%03i.txt" % dataset_ind, 'a')
    cosmo = FlatLambdaCDM(Om0 = 0.3, H0 = true_H0)


    
    for i in range(nsne):

        f = open(p_wd + "/params_%04i.dat" % i, 'w')
        for key in all_SNe[i]:
            f.write(key + "  " + str(all_SNe[i][key]) + '\n')
        f.write("x0 " + str(SNe_x0s[i]) + '\n')
        f.write("peak_mag " + str(peak_mags[i]) + '\n')
        f.write("observed  " + str(observed_SNe[i]) + '\n')
        
        f.close()
        
        

        #phases = (dates - params[i]["t0"])/(1. + params[i]["z"])

        if observed_SNe[i]:
            SN_name = "SN%s%s%04i" % (z_range_key, all_SNe[i]["outlier"]*"O" + (1 - all_SNe[i]["outlier"])*"I", i)
            this_wd = wd + "/" + SN_name
            do_it("mkdir -p " + this_wd)


            f = open(this_wd + "/lightfile", 'w')
            f.write("z_cmb  %f\n" % all_SNe[i]["z"])
            f.write("z_heliocentric  %f\n" % all_SNe[i]["z"])
            f.write("MWEBV 0.0\n")
            f.write("RA  0.0\n")
            f.write("DEC  0.0\n")
            f.write("Mass  %f  -0.03  0.03\n" % all_SNe[i]["mass"])
            f.close()
            
            if all_SNe[i]["z"] < 0.01:
                # If calibrator
                f_ladder.write(SN_name + "  " + str(cosmo.distmod(all_SNe[i]["z"]).value) + '\n')


            model = get_SNCosmo_model(all_SNe[i], source)

            these_dates = dates[np.where(np.abs(dates - all_SNe[i]["t0"]) < 50*(1 + all_SNe[i]["z"]))]
            
            for band in bands_to_use:
                rest_frame_band = sncosmo.Bandpass(dict_of_obsframe_filt[band].wave/(1 + all_SNe[i]["z"]), dict_of_obsframe_filt[band].trans)

                  
                try:
                    fluxes = model.bandflux(band, these_dates, zp = 27.5, zpsys = "ab")
                    if np.any(np.isnan(fluxes)):
                        print("NaN found!", list(fluxes))
                        raise_time
                    good_band = 1
                except:
                    print("Couldn't get band", all_SNe[i]["z"], band)
                    good_band = 0
                
                    
                #print(fluxes)
                #print(fluxcov)
                if good_band:
                    
                    source.set(x1 = all_SNe[i]["x1"], c = all_SNe[i]["c"])
                    r_cov = source.bandflux_rcov(band = np.array([rest_frame_band]*len(these_dates)),
                                                 phase = (these_dates - all_SNe[i]["t0"])/(1. + all_SNe[i]["z"]))

                    r_cov = np.clip(r_cov, -1, 1)

                    fluxcov = np.outer(fluxes, fluxes)*r_cov
                    #rcov = source.rcov_()
                    #fluxes, fluxcov = model.bandfluxcov(band, all_SNe[i]["t0"], zp = 27.5, zpsys = "ab") # For SALT2, not SALT3
                    

                    model_fluxes = fluxes + np.random.multivariate_normal(mean = fluxes*0.,
                                                                          cov = fluxcov*(opts.modeluncertainty*0.9999 + 0.0001))
                        
                    obs_fluxes = model_fluxes + np.random.normal(size = len(model_fluxes))*obs_err*opts.addnoise
                    
                    f = open(this_wd + "/lc2fit_" + band + ".dat", 'w')
                    f.write("""#Date :
#Flux :
#Fluxerr :
#ZP :
#end :
@INSTRUMENT """ + band_to_instr(band) + """
@BAND """ + band_to_name(band) + """
@MAGSYS AB
""")
                    for j in range(len(obs_fluxes)):
                        towrite = [dates[j], obs_fluxes[j], obs_err*(opts.addnoise*0.99 + 0.01), 27.5 + cal_offsets[band]]
                        towrite = [str(item) for item in towrite]
                        f.write("  ".join(towrite) + '\n')
                    f.close()
            
    f_ladder.close()

def set_up_UNITY(wd, dataset_ind, oneDint, nocal, noselection, twopop, include_low, cosmomodel, distance_ladder_fl, two_x1):
    dataset_list = ["../dataset_S_%03i_v1.txt" % dataset_ind]*include_low*(opts.zrangekeys.count("S") > 0) + ["../dataset_L_%03i_v1.txt" % dataset_ind]*include_low*(opts.zrangekeys.count("L") > 0) + ["../dataset_H_%03i_v1.txt" % dataset_ind] + ["../dataset_V_%03i_v1.txt" % dataset_ind]*include_low
    dataset_list = str(dataset_list).replace(" ", "")

    if noselection:
        # No selection effects
        if twopop:
            if include_low:
                population_model = "a 3"
            else:
                population_model = "a 2"
        else:
            # No selection effects, one pop.
            population_model = "sample 0.5"
    else:
        if include_low:
            population_model = "sample 0.5 0.0 1.0"
        else:
            population_model = "sample 0.5"

    if cosmomodel == 5:
        fix_Om = "0.3"
    else:
        fix_Om = "0"
            

    f = open(wd + "paramfile.txt", 'w')
    f.write("""
do_blind		0
filenamelist		%s

weird_sn_list		"../weird_sn_list.txt"
mag_cut			%s
stan_code		%s
sample_file		"None"
calibration_uncertainties		%s
distance_ladder         %s

min_redshift		0.01
max_redshift		3.
max_firstphase		100.
min_lastphase		-100.
max_color_uncertainty	0.2
max_color		0.3
max_MWEBV		0.3
min_color		-0.3
remap_x1		[0.,0.]


# Units of c:
pec_vel_disp		0.001
# Units of magnitudes per redshift
lensing_disp		0.055
MWEBV_zeropoint_EBV	0.0001
outl_frac		0.02
redshift_coeff_type     %s
electron_coeff		[0.000042,0.0000042]
IG_extinction_coeff	0.0001


max_params_to_save	50


do_twoalphabeta		1
threeD_unexplained	%i

    
iter			2500
n_jobs			4
chains			4

do_host_mass		1
fix_Om			%s
MB_by_sample		0
include_pec_cov		0
separate_mass_x1c	1
""" % (dataset_list,
       '"../mag_cuts.txt"'*(params["obs_mag_selection"]) + '"../mag_cuts_x0.txt"'*(1 - params["obs_mag_selection"]),
       '"$UNITY/scripts/stan_code_H0.txt"'*(1 - noselection)*(two_x1 == 0) + '"$UNITY/scripts/stan_code_x12.txt"'*(1 - noselection)*(two_x1 == 1) + '"$UNITY/scripts/stan_code_simple_no_sel.txt"'*noselection,
       '"../calibration_uncertainties.txt"'*(1 - nocal) + '"../calibration_uncertainties_small.txt"'*nocal,
       distance_ladder_fl,
       population_model, 1 - oneDint, fix_Om))
    f.close()

    f = open(wd + "run.sh", 'w')
    f.write("""#!/bin/bash
#SBATCH --job-name=example
#SBATCH --partition=shared
#SBATCH --time=0-""" + str(int(5 + 6*opts.nvisit/200 + 6*include_low*opts.nvisit/200)) + """:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=""" + str(int((10 + 10*include_low)*opts.nvisit/200)) +  """G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
export UNION=../
""")
    f.write("cd " + pwd + '\n')
    f.write("cd " + wd + '\n')

    f.write("~/.conda/envs/py39/bin/python $UNITY/scripts/read_and_sample_H0.py paramfile.txt %i > log.txt\n" % cosmomodel)

    f.close()

pwd = do_it("pwd")


parser = argparse.ArgumentParser()
parser.add_argument('--ndataset', help='N datasets', type = int)
parser.add_argument('--addnoise', help='Add Noise', type = int)
parser.add_argument('--addcalibration', help='Add Calibration', type = int)
parser.add_argument('--modeluncertainty', help='Add Model Uncertainty', type = int)
parser.add_argument('--prefixname', help='Prefix Name for Directory', type = str)
parser.add_argument('--skewdist', help='x1 and c distributions have skew', type = int)
parser.add_argument('--volumelimited', help='volume-limited, not magnitude-limited datasets', type=int)
parser.add_argument('--obsmagselection', help="If magnitude-limited, select based on observer-frame magntiudes, not rest-frame x0", type=int)
parser.add_argument('--zrangekeys', help="SLHV for very low-z, low-z SDSS, high-z SDSS, very high-z HST", type=str)
parser.add_argument('--nnearbyperset', help = "Number of nearby SNe for S and L not counting calibrators", type=int)
parser.add_argument('--ncalibperset', help = "Number of calibrator SNe for sets S and L", type=int)
parser.add_argument('--sigzp', help="Zeropoint Uncertainty Size", type=float)
parser.add_argument('--nvisit', help="Number of visits", type=int, default = 200)
parser.add_argument('--sigmabetaR', help="Scatter in beta_R", type=float)
parser.add_argument("--sigunexplained", help="Unexplained dispersion", type=float, default=0.12)
parser.add_argument("--simtype", help="Union3 or Union3.1", type=str)



opts = parser.parse_args()


salt2_version = "salt3-f22"
source = sncosmo.SALT3Source(modeldir = os.environ["PATHMODEL"] + "/" + salt2_version + "/")

nonSALTkeys = ["latentmu", "MB", "mass", "delta_mBx1c", "latentMB", "latentx1", "latentc", "latentcB", "latentcR", "beta_R", "delta_mB", "delta_x1", "delta_c", "delta_mu", "outlier", "x1_slow"]

dict_of_obsframe_filt = {}
for filt in "griz":
    dict_of_obsframe_filt["sdss" + filt] = sncosmo.get_bandpass("sdss" + filt)
for filt in ["f775w", "f850lp", "f105w", "f125w", "f160w"]:
    dict_of_obsframe_filt[filt] = sncosmo.get_bandpass(filt)
for filt in "ubv":
    dict_of_obsframe_filt["uvot::" + filt] = sncosmo.get_bandpass("uvot::" + filt)
if opts.zrangekeys.count("S"):
    for filt in "UBVRI":
        lambs = np.arange(2900., 11000.)
        tmp_trans = Spectra.Spectra(instrument = "OtherNearby", band = "Bessell12_" + filt).transmission_fn(lambs)
        inds = np.where(tmp_trans > tmp_trans.max()/1e4)[0]
        
        dict_of_obsframe_filt["Bessell12_" + filt] = sncosmo.Bandpass(lambs[inds[0]: inds[-1]],
                                                                      tmp_trans[inds[0]: inds[-1]])

        sncosmo.register(dict_of_obsframe_filt["Bessell12_" + filt], "Bessell12_" + filt)
        
[tmpx, tmpy] = readcol(os.environ["UNITY"] + "/simulated/SN_rates.txt", 'ff')
SN_rate_function = interp1d(tmpx, tmpy*1e-4, kind = 'linear')


if opts.simtype == "Union3":
    params = dict(salt2_version = salt2_version, n_visit = opts.nvisit, nnearbyperset = opts.nnearbyperset, ncalibperset = opts.ncalibperset,
                  ndeg2 = 10., nsnepernight = 3, ndataset = opts.ndataset, cadence = 4., HST_cadence = 17., HST_visit = 6.,
                  obs_mag_selection = opts.obsmagselection, volume_limited = opts.volumelimited, modeluncertainty = opts.modeluncertainty,
                  Rx1 = 0.5 + 0.45*(1 - opts.skewdist), tau_x1 = -0.8*opts.skewdist,
                  Rc = 0.05 + 0.035*(1 - opts.skewdist), tau_c = 0.07*opts.skewdist,
                  tot_sig_unexplained = opts.sigunexplained, sigma_int_fast = 0.,
                  alpha = 0.15, sigma_beta_R = opts.sigmabetaR,
                  beta_B = 3.1, beta_R = 3.1, delta_beta_R = 0., delta = 0.08, MB = -19.1,
                  step_width = 0., #0.15,
                  outlierfrac = 0.02, sigzp = opts.sigzp, true_H0 = true_H0)
elif opts.simtype == "Union3.1":
    params = dict(salt2_version = salt2_version, n_visit = opts.nvisit, nnearbyperset = opts.nnearbyperset, ncalibperset = opts.ncalibperset,
                  ndeg2 = 10., nsnepernight = 3, ndataset = opts.ndataset, cadence = 4., HST_cadence = 17., HST_visit = 6.,
                  obs_mag_selection = opts.obsmagselection, volume_limited = opts.volumelimited, modeluncertainty = opts.modeluncertainty,
                  Rx1_fast = 0.8, Rx1_slow = 0.5,
                  x1_star_fast = -0.9, x1_star_slow = 0.5,
                  Rc_fast = 0.06, Rc_slow = 0.04,
                  c_star_fast = -0.05, c_star_slow = -0.08,
                  frac_x1_slow_high = 0.5, frac_x1_slow_low = 0.8,
                  
                  tau_c = 0.07*opts.skewdist,
                  tot_sig_unexplained = opts.sigunexplained, sigma_int_fast = 0.08, alpha = 0.15, sigma_beta_R = opts.sigmabetaR,
                  beta_B = 2.1, beta_R = 3.8, delta_beta_R = 1.2, delta = 0.0, MB = -19.1, MB_fast_minus_slow = -0.14,
                  step_width = 0., #0.15,
                  outlierfrac = 0.02, sigzp = opts.sigzp, true_H0 = true_H0)
else:
    assert "Unknown type!" + str(opts.simtype)
    
do_it("rm -fr " + opts.prefixname)
do_it("mkdir " + opts.prefixname)

if params["volume_limited"]:
    nominal_mag_limits = dict(S = 19.0, L = 19.0, H = 24.0, V = 26.0)
else:
    nominal_mag_limits = dict(S = 16.0, L = 18.0, H = 23.0, V = 26.0)

f = open(opts.prefixname + "/mag_cuts.txt", 'w')
for dataset_ind in range(opts.ndataset):
    for outl in ["", "_nooutl"]:
        f.write("dataset_S_%03i%s_v1.txt  $UNITY/paramfiles/sdssr_selection_salt3-f22_ab.txt    %.2f            0.5\n" % (dataset_ind, outl, nominal_mag_limits["S"]))
        f.write("dataset_L_%03i%s_v1.txt  $UNITY/paramfiles/sdssr_selection_salt3-f22_ab.txt    %.2f            0.5\n" % (dataset_ind, outl, nominal_mag_limits["L"]))
        f.write("dataset_H_%03i%s_v1.txt  $UNITY/paramfiles/sdssi_selection_salt3-f22_ab.txt    %.2f            0.5\n" % (dataset_ind, outl, nominal_mag_limits["H"]))
        f.write("dataset_V_%03i%s_v1.txt  $UNITY/paramfiles/f125w_selection_salt3-f22_ab.txt	%.2f		0.25\n" % (dataset_ind, outl, nominal_mag_limits["V"]))
f.close()

f = open(opts.prefixname + "/mag_cuts_x0.txt", 'w')
for dataset_ind in range(opts.ndataset):
    f.write("dataset_S_%03i_v1.txt  $UNITY/paramfiles/No_k_correct.txt    %.2f            0.5\n" % (dataset_ind, nominal_mag_limits["S"]))
    f.write("dataset_L_%03i_v1.txt  $UNITY/paramfiles/No_k_correct.txt    %.2f            0.5\n" % (dataset_ind, nominal_mag_limits["L"]))
    f.write("dataset_H_%03i_v1.txt  $UNITY/paramfiles/No_k_correct.txt    %.2f            0.5\n" % (dataset_ind, nominal_mag_limits["H"]))
    f.write("dataset_V_%03i_v1.txt  $UNITY/paramfiles/No_k_correct.txt    %.2f            0.5\n" % (dataset_ind, nominal_mag_limits["V"]))
f.close()


f = open(opts.prefixname + "/calibration_uncertainties.txt", 'w')
f.write("""
('Fundamental', (3000.0, 4000.0)):                                                                      0.0001
('Fundamental', (4000.0, 5000.0)):                                                                      0.0001
('Fundamental', (6000.0, 8000.0)):                                                                      0.0001
('Fundamental', (8000.0, 100000.0)):                                                                    0.0001
('Fundamental', (10000.0, 100000.0)):                                                                   0.0001


"SALT_UV_CAL":                                                                                          0.0001
"SALT_U_CAL":                                                                                           0.0001
"SALT_I_CAL":                                                                                           0.0001

('Zeropoint', 'UVOT|UVOT_u'): """ + str(opts.sigzp) + """
('Zeropoint', 'UVOT|UVOT_b'): """ + str(opts.sigzp) + """
('Zeropoint', 'UVOT|UVOT_v'): """ + str(opts.sigzp) + """
('Lambda', 'UVOT|UVOT_u'):    0.01
('Lambda', 'UVOT|UVOT_b'):    0.01
('Lambda', 'UVOT|UVOT_v'):    0.01

('Zeropoint', 'OtherNearby|Bessell12_U'): """ + str(opts.sigzp) + """
('Zeropoint', 'OtherNearby|Bessell12_B'): """ + str(opts.sigzp) + """
('Zeropoint', 'OtherNearby|Bessell12_V'): """ + str(opts.sigzp) + """
('Zeropoint', 'OtherNearby|Bessell12_R'): """ + str(opts.sigzp) + """
('Zeropoint', 'OtherNearby|Bessell12_I'): """ + str(opts.sigzp) + """

('Lambda', 'OtherNearby|Bessell12_U'):     0.01
('Lambda', 'OtherNearby|Bessell12_B'):     0.01
('Lambda', 'OtherNearby|Bessell12_V'):     0.01
('Lambda', 'OtherNearby|Bessell12_R'):     0.01
('Lambda', 'OtherNearby|Bessell12_I'):     0.01


('Zeropoint', 'SDSS|SDSS_u'): """ + str(opts.sigzp) + """
('Zeropoint', 'SDSS|SDSS_g'): """ + str(opts.sigzp) + """
('Zeropoint', 'SDSS|SDSS_r'): """ + str(opts.sigzp) + """
('Zeropoint', 'SDSS|SDSS_i'): """ + str(opts.sigzp) + """
('Zeropoint', 'SDSS|SDSS_z'): """ + str(opts.sigzp) + """
('Lambda', 'SDSS|SDSS_u'):    0.01
('Lambda', 'SDSS|SDSS_g'):    0.01
('Lambda', 'SDSS|SDSS_r'):    0.01
('Lambda', 'SDSS|SDSS_i'):    0.01
('Lambda', 'SDSS|SDSS_z'):    0.01

('Zeropoint', 'ACSWF|F775W'): """ + str(opts.sigzp) + """
('Zeropoint', 'ACSWF|F850LP'): """ + str(opts.sigzp) + """
('Zeropoint', 'WFC3|WFC3_f105w'): """ + str(opts.sigzp) + """
('Zeropoint', 'WFC3|WFC3_f125w'): """ + str(opts.sigzp) + """
('Zeropoint', 'WFC3|WFC3_f160w'): """ + str(opts.sigzp) + """
('Lambda', 'ACSWF|F775W'): 0.01
('Lambda', 'ACSWF|F850LP'): 0.01
('Lambda', 'WFC3|WFC3_f105w'): 0.01
('Lambda', 'WFC3|WFC3_f125w'): 0.01
('Lambda', 'WFC3|WFC3_f160w'): 0.01

""")
f.close()


f = open(opts.prefixname + "/calibration_uncertainties_small.txt", 'w')
f.write("""
('Fundamental', (3000.0, 4000.0)):                                                                      0.0001
('Fundamental', (4000.0, 5000.0)):                                                                      0.0001
('Fundamental', (6000.0, 8000.0)):                                                                      0.0001
('Fundamental', (8000.0, 100000.0)):                                                                    0.0001
('Fundamental', (10000.0, 100000.0)):                                                                   0.0001


"SALT_UV_CAL":                                                                                          0.0001
"SALT_U_CAL":                                                                                           0.0001
"SALT_I_CAL":                                                                                           0.0001

('Zeropoint', 'UVOT|UVOT_u'): 0.0001
('Zeropoint', 'UVOT|UVOT_b'): 0.0001
('Zeropoint', 'UVOT|UVOT_v'): 0.0001
('Lambda', 'UVOT|UVOT_u'):    0.01
('Lambda', 'UVOT|UVOT_b'):    0.01
('Lambda', 'UVOT|UVOT_v'):    0.01

('Zeropoint', 'SDSS|SDSS_u'): 0.0001
('Zeropoint', 'SDSS|SDSS_g'): 0.0001
('Zeropoint', 'SDSS|SDSS_r'): 0.0001
('Zeropoint', 'SDSS|SDSS_i'): 0.0001
('Zeropoint', 'SDSS|SDSS_z'): 0.0001
('Lambda', 'SDSS|SDSS_u'):    0.01
('Lambda', 'SDSS|SDSS_g'):    0.01
('Lambda', 'SDSS|SDSS_r'):    0.01
('Lambda', 'SDSS|SDSS_i'):    0.01
('Lambda', 'SDSS|SDSS_z'):    0.01

('Zeropoint', 'ACSWF|F775W'): 0.0001
('Zeropoint', 'ACSWF|F850LP'): 0.0001
('Zeropoint', 'WFC3|WFC3_f105w'): 0.0001
('Zeropoint', 'WFC3|WFC3_f125w'): 0.0001
('Zeropoint', 'WFC3|WFC3_f160w'): 0.0001
('Lambda', 'ACSWF|F775W'): 0.01
('Lambda', 'ACSWF|F850LP'): 0.01
('Lambda', 'WFC3|WFC3_f105w'): 0.01
('Lambda', 'WFC3|WFC3_f125w'): 0.01
('Lambda', 'WFC3|WFC3_f160w'): 0.01


""")
f.close()



f = open(opts.prefixname + "/weird_sn_list.txt", 'w')
f.close()


f = open(opts.prefixname + "/check_Om.sh", 'w')
f.write("""#!/bin/bash
#SBATCH --job-name=checkruns
#SBATCH --partition=shared
#SBATCH --time=0-18:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile

for i in {1..72}
do
    cd """ + opts.prefixname + """
    python $UNITY/simulated/step1B_Om_sampled_well.py 1 */log.txt

    sleep 900
done
""")


f_UNITY = []

ever_include_low = opts.zrangekeys.count("L") or opts.zrangekeys.count("S")

for include_low in [0] + [1]*ever_include_low:
    f_UNITY.append(open(opts.prefixname + "/run_UNITY" + "_low"*include_low + ".sh", 'w'))

    f_UNITY[include_low].write("""#!/bin/bash
#SBATCH --job-name=runU
#SBATCH --partition=shared
#SBATCH --time=0-01:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G # Memory per node my job requires
#SBATCH --error=runU-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=runU-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
""")
    f_UNITY[include_low].write("cd " + pwd + '\n')
    f_UNITY[include_low].write("cd " + opts.prefixname + '\n')
    f_UNITY[include_low].write("~/.conda/envs/py39/bin/python $PATHMODEL/python_code/cut_fits.py dataset*\n")


f_interleave = open(opts.prefixname + "/run_interleave.sh", 'w')
f_interleave.write("""#!/bin/bash
#SBATCH --job-name=runU
#SBATCH --partition=shared
#SBATCH --time=1-12:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G # Memory per node my job requires
#SBATCH --error=runU-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=runU-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile

cd """ + opts.prefixname + """
sbatch check_Om.sh
""")

    
for dataset_ind in tqdm.trange(opts.ndataset):
    cal_offsets = {}
    for band in dict_of_obsframe_filt:
        cal_offsets[band] = np.random.normal()*opts.sigzp*opts.addcalibration

    frac_var_mBx1c = np.random.exponential(scale=1.0, size=3)
    frac_var_mBx1c /= sum(frac_var_mBx1c)

    params["delta_h"] = np.random.random()
    params["frac_var_mBx1c"] = frac_var_mBx1c
    
    params["sigma_unexplained_3d"] = [params["tot_sig_unexplained"]*np.sqrt(params["frac_var_mBx1c"][0]),
                                      params["tot_sig_unexplained"]*np.sqrt(params["frac_var_mBx1c"][1])/0.14,
                                      params["tot_sig_unexplained"]*np.sqrt(params["frac_var_mBx1c"][2])/3.]

    params["sigma_unexplained_3d_slow"] = params["sigma_unexplained_3d"]
    
    params["sigma_unexplained_3d_fast"] = [np.sqrt(params["tot_sig_unexplained"]**2. + params["sigma_int_fast"]**2.)*np.sqrt(params["frac_var_mBx1c"][0]),
                                           np.sqrt(params["tot_sig_unexplained"]**2. + params["sigma_int_fast"]**2.)*np.sqrt(params["frac_var_mBx1c"][1])/0.14,
                                           np.sqrt(params["tot_sig_unexplained"]**2. + params["sigma_int_fast"]**2.)*np.sqrt(params["frac_var_mBx1c"][2])/3.]

    
    assert np.isclose(params["tot_sig_unexplained"], np.sqrt(
        params["sigma_unexplained_3d"][0]**2.
        + (0.14*params["sigma_unexplained_3d"][1])**2.
        + (3.0*params["sigma_unexplained_3d"][2])**2.
    ))

    f = open(opts.prefixname + "/params_%03i.dat" % dataset_ind, 'w')
    for param in params:
        f.write(param + "  " + str(params[param]) + '\n')
    f.close()

    for z_range_key in opts.zrangekeys:
        wd = opts.prefixname + "/dataset_%s_%03i/" % (z_range_key, dataset_ind)
        do_it("mkdir " + wd)
        make_dataset(wd, cal_offsets = cal_offsets, dataset_ind = dataset_ind)


    
    [ladder_SN, ladder_mu] = readcol(opts.prefixname + "/distance_ladder_true_vals_%03i.txt" % dataset_ind, 'af')

    for common_unc in [0.0, 0.02]:
        for sigma_ladder_unexpl in [0.0, 0.05]:
            common_offset = np.random.normal()*common_unc
            sigma_ladder_total = np.sqrt(sigma_ladder_unexpl**2. + 0.05**2.)
            
            f_ladder = open(opts.prefixname + "/distance_ladder_obs_vals_common=%.3f_unexp=%.3f_%03i.txt" % (common_unc, sigma_ladder_unexpl, dataset_ind), 'w')
            
            for lad_ind in range(len(ladder_SN)):
                f_ladder.write(ladder_SN[lad_ind] + "  " + str(ladder_mu[lad_ind] + np.random.normal()*sigma_ladder_total + common_offset) + "  0.05  " + str(common_unc) + '\n') # Always write 0.05, unexplained is separate
            f_ladder.close()
    


    f_interleave.write("cd " + pwd + '\n')
    f_interleave.write("cd " + opts.prefixname + '\n')
    f_interleave.write("~/.conda/envs/py39/bin/python $PATHMODEL/python_code/cut_fits.py dataset_?_%03i | tail \n" % dataset_ind)

    if opts.ncalibperset == 0:
        # Test dark energy
        for include_low in [1]: #[0] + [1]*ever_include_low:
            for oneDint, nocal, noselection, twopop, two_x1 in ([0, 0, 0, 0, 0], [0, 0, 0, 0, 1]): #[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]): # [0, 1, 0, 0]
                for cosmomodel in [1]*(1 - include_low) + [5]*include_low:
                    wd = opts.prefixname + "/UNITY%s%s%s%s%s%s%s_%03i/" % ("L"*include_low + "H", "_1D"*oneDint,
                                                                           "_nocal"*nocal, "_nosel"*noselection, "_twopop"*twopop, "_twox1"*two_x1, "_cos=" + str(cosmomodel), dataset_ind)
                    do_it("mkdir " + wd)

                    set_up_UNITY(wd, dataset_ind = dataset_ind, oneDint = oneDint, nocal = nocal, noselection = noselection, twopop = twopop, include_low = include_low, cosmomodel = cosmomodel, distance_ladder_fl = "None", two_x1 = two_x1)

                    f_UNITY[include_low].write("cd " + pwd + '\n')
                    f_UNITY[include_low].write("cd " + wd + '\n')
                    f_UNITY[include_low].write("sbatch run.sh\n")

                    f_interleave.write("cd " + pwd + '\n')
                    f_interleave.write("cd " + wd + '\n')
                    f_interleave.write("sbatch run.sh\n")
    else:
        # Test Hubble constant
        include_low = 1
        oneDint = 0
        nocal = 0
        noselection = 0
        twopop = 0
        cosmomodel = 1
        for two_x1 in [0, 1]:
        
            wd = opts.prefixname + "/UNITY%s%s%s%s%s%s%s_%03i/" % ("L"*include_low + "H", "_1D"*oneDint,
                                                                   "_nocal"*nocal, "_nosel"*noselection, "_twopop"*twopop, "twox1"*two_x1, "_cos=" + str(cosmomodel), dataset_ind)
            do_it("mkdir " + wd)
            
            set_up_UNITY(wd, dataset_ind = dataset_ind, oneDint = oneDint, nocal = nocal, noselection = noselection, twopop = twopop, include_low = include_low, cosmomodel = cosmomodel, two_x1=two_x1,
                         distance_ladder_fl = "../distance_ladder_obs_vals_common=%.3f_unexp=%.3f_%03i.txt" % (0.02, 0.0, dataset_ind))

            f_UNITY[include_low].write("cd " + pwd + '\n')
            f_UNITY[include_low].write("cd " + wd + '\n')
            f_UNITY[include_low].write("sbatch run.sh\n")
            
            f_interleave.write("cd " + pwd + '\n')
            f_interleave.write("cd " + wd + '\n')
            f_interleave.write("sbatch run.sh\n")

        
        
f_UNITY[0].close()
f_UNITY[1].close()
f_interleave.close()

print("Don't forget to set -wr = 0 when doing LC fitting!")
