from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
import numpy as np
from FileRead import readcol
from scipy.interpolate import interp1d
import sncosmo
from astropy.table import Table
import subprocess
import tqdm
import pickle
import os
import sys
from astropy.cosmology import FlatLambdaCDM


def get_SNCosmo_model(these_params, source):
    #sncosmo_model = sncosmo.Model(source="salt2-extended")
    sncosmo_model = sncosmo.Model(source=source)

    tmp_params = dict(these_params)
    for nonSALTkey in nonSALTkeys:
        del tmp_params[nonSALTkey]
    sncosmo_model.set(**tmp_params)

    
    cosmo = FlatLambdaCDM(Om0 = 0.3, H0 = 70.)
    ten_pc_z = 2.33494867e-9
    assert abs(cosmo.distmod(z=ten_pc_z).value) < 1.e-3, "Distance modulus zeropoint wrong!"

    ampl = 1.
    for i in range(2):
        sncosmo_model.set(z=ten_pc_z, t0=0., x0=ampl)

        mag = sncosmo_model.bandmag('bessellb', 'ab', 0.)
        #print "mag, ampl ", mag, ampl
        ampl *= 10.**(0.4*(mag - these_params["MB"]))
        
    mu = cosmo.distmod(these_params["z"]).value
    #print "mu ", mu

    sncosmo_model.set(z=these_params["z"], t0=these_params["t0"], x0 = ampl*10**(-0.4*mu))
    return sncosmo_model


def approxmB(model, date):
    x0 = model.get("x0")
    t0 = model.get("t0")

    return -2.5*np.log10(x0 * np.exp(  -0.5*((date - t0)/10.)**2.  )
                         )


def make_dataset(wd, cal_offsets):
    is_low_z = wd.count("_L_")
    if is_low_z:
        assert wd.count("_H_") == 0

    if is_low_z:
        obs_err = 80. # Depth of 21.0 at 5 sigma
    else:
        obs_err = 5. # Depth of 24.0 at 5 sigma
        

    dates = np.arange(params["n_visit"], dtype=np.float64)*params["cadence"]

    model = sncosmo.Model(source=source)

    if is_low_z:
        zlist = list(sncosmo.zdist(0., zmax = 0.1, time=dates[-1] - dates[0] - 4*params["cadence"], area=500.))
    else:
        zlist = list(sncosmo.zdist(0., zmax = 1.0, time=dates[-1] - dates[0] - 4*params["cadence"], area=params["ndeg2"]))
    print("len(zlist)", len(zlist))

    nsne = len(zlist)
    
    all_SNe = []
    for z in zlist:
        p = dict(z = z,
                 t0 = np.random.uniform(dates[0] + params["cadence"]*2, dates[-1] - params["cadence"]*2),
                 x1 = np.random.normal()*params["Rx1"] + (np.random.exponential() - 1.)*params["tau_x1"],
                 c = np.random.normal()*params["Rc"] + (np.random.exponential() - 1.)*params["tau_c"],
                 delta_m = np.random.normal()*np.sqrt(params["gray_sig_unexplained"]**2. + (0.055*z)**2.),
                 mass = 10. + np.random.normal())


        relative_step_z = 1.9/(1. + 0.9*10.**(0.95*p["z"]))
        relative_step_z = relative_step_z*(1 - params["delta_h"]) + params["delta_h"]

        delta_z = params["delta"]*(relative_step_z*params["delta_h"] + params["delta_h"])
        mabs = params["MB"] + p["delta_m"] - params["alpha"]*p["x1"] + 3.1*p["c"] - params["delta"]*(p["mass"] > 10.)*relative_step_z
        p["MB"] = mabs

        all_SNe.append(p)



    observed_SNe = np.zeros(nsne, dtype=np.int16)
    
    for night in dates:
        all_mags = []
        
        for i in range(nsne):
            if observed_SNe[i] == 0 and np.abs(night - all_SNe[i]["t0"]) < 50:
                model = get_SNCosmo_model(all_SNe[i], source)
                if params["obs_mag_selection"]:
                    if is_low_z:
                        all_mags.append(model.bandmag("sdssr", "ab", night))
                    else:
                        all_mags.append(model.bandmag("sdssi", "ab", night))
                else:
                    all_mags.append(approxmB(model, night))
            else:
                all_mags.append(1e20)
                
        all_mags = np.array(all_mags)
        print("all_mags", all_mags)
                                
        inds = np.argsort(all_mags)

        for i in range(params["nsnepernight"]):
            observed_SNe[inds[i]] = 1

    
    plt.hist([all_SNe[i]["z"] for i in range(nsne) if observed_SNe[i] == 1], alpha = 0.5, label = str(sum(observed_SNe)))
    plt.hist([all_SNe[i]["z"] for i in range(nsne) if observed_SNe[i] == 0], alpha = 0.5, label = str(nsne - sum(observed_SNe)))
    plt.legend(loc = 'best')
    plt.savefig("selected.pdf")
    plt.close()

    peak_mags = []
    SNe_x0s = []
    
    for i in range(nsne):
        model = get_SNCosmo_model(all_SNe[i], source)
        if params["obs_mag_selection"]:
            if is_low_z:
                peak_mags.append(model.bandmag("sdssr", "ab", all_SNe[i]["t0"]))
            else:
                peak_mags.append(model.bandmag("sdssi", "ab", all_SNe[i]["t0"]))
        else:
            peak_mags.append(approxmB(model, all_SNe[i]["t0"]))
        SNe_x0s.append(model.get("x0"))
        
    peak_mags = np.array(peak_mags)
    plt.hist(peak_mags[np.where(observed_SNe)], alpha = 0.5)
    plt.hist(peak_mags[np.where(1 - observed_SNe)], alpha = 0.5)
    plt.savefig("selected_mags.pdf")
    plt.close()
    

    print("total observed_SNe", sum(observed_SNe))


    p_wd = wd.replace("dataset_", "UNITY_") + "/SN_params/"
    subprocess.getoutput("mkdir -p " + p_wd)

    
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
            this_wd = wd + "/SN%04i" % i
            subprocess.getoutput("mkdir -p " + this_wd)


            f = open(this_wd + "/lightfile", 'w')
            f.write("z_cmb  %f\n" % all_SNe[i]["z"])
            f.write("z_heliocentric  %f\n" % all_SNe[i]["z"])
            f.write("MWEBV 0.0\n")
            f.write("RA  0.0\n")
            f.write("DEC  0.0\n")
            f.write("Mass  %f  -0.05  0.05\n" % all_SNe[i]["mass"])
            f.close()
            

            model = get_SNCosmo_model(all_SNe[i], source)

            these_dates = dates[np.where(np.abs(dates - all_SNe[i]["t0"]) < 50*(1 + all_SNe[i]["z"]))]
            
            for band in ['sdssg', 'sdssr', 'sdssi', 'sdssz']:
                rest_frame_band = sncosmo.Bandpass(SDSS_obs_frame[band[-1]].wave/(1 + all_SNe[i]["z"]), SDSS_obs_frame[band[-1]].trans)

                  
                try:
                    fluxes = model.bandflux(band, these_dates, zp = 27.5, zpsys = "ab")

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
                    

                    model_fluxes = np.random.multivariate_normal(mean = fluxes, cov = fluxcov*(add_noise_and_calibration*0.999 + 0.001))
                    obs_fluxes = model_fluxes + np.random.normal(size = len(model_fluxes))*obs_err*add_noise_and_calibration
                    
                    f = open(this_wd + "/lc2fit_" + band + ".dat", 'w')
                    f.write("""#Date :
        #Flux :
        #Fluxerr :
        #ZP :
        #end :
        @INSTRUMENT SDSS
        @BAND SDSS_""" + band[-1]  + """
        @MAGSYS AB
        """)
                    for j in range(len(obs_fluxes)):
                        towrite = [dates[j], obs_fluxes[j], obs_err*(add_noise_and_calibration*0.99 + 0.01), 27.5 + cal_offsets[band[-1]]*add_noise_and_calibration]
                        towrite = [str(item) for item in towrite]
                        f.write("  ".join(towrite) + '\n')
                    f.close()
            

def set_up_UNITY(wd, dataset_ind, oneDint, nocal, noselection, twopop, include_low):
    dataset_list = ["../dataset_L_%03i_v1.txt"]*include_low + ["../dataset_H_%03i_v1.txt"]
    dataset_list = str(dataset_list)

    if twopop:
        population_model = "a 2"
    else:
        if include_low:
            population_model = "sample 0.5 0.0"
        else:
            population_model = "sample 0.5"


    f = open(wd + "paramfile.txt", 'w')
    f.write("""
do_blind		0
filenamelist		%s

weird_sn_list		"../weird_sn_list.txt"
mag_cut			%s
stan_code		%s
sample_file		"None"
calibration_uncertainties		%s


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
electron_coeff		[0.0042,0.00042]
IG_extinction_coeff	0.0001

    
do_twoalphabeta		1
threeD_unexplained	%i

    
iter			2500
n_jobs			4
chains			4

do_host_mass		1
fix_Om			0
MB_by_sample		0
include_pec_cov		0
separate_mass_x1c	1
    """ % (dataset_list,
           '"../mag_cuts.txt"'*(params["obs_mag_selection"]) + '"../mag_cuts_x0.txt"'*(1 - params["obs_mag_selection"]),
           '"$UNITY/scripts/stan_code_simple.txt"'*(1 - noselection) + '"$UNITY/scripts/stan_code_simple_no_sel.txt"'*noselection,
           '"../calibration_uncertainties.txt"'*(1 - nocal) + '"../calibration_uncertainties_small.txt"'*nocal,
           population_model, 1 - oneDint))
    f.close()

    f = open(wd + "run.sh", 'w')
    f.write("""#!/bin/bash
#SBATCH --job-name=example
#SBATCH --partition=shared
#SBATCH --time=0-""" + str(12 + 6*include_low) + """:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
export UNION=../
""")
    f.write("cd " + pwd + "/" + wd + '\n')
    f.write("python $UNITY/scripts/read_and_sample.py paramfile.txt 1 > log.txt\n")

    f.close()

pwd = subprocess.getoutput("pwd")


ndataset = int(sys.argv[1])
add_noise_and_calibration = float(sys.argv[2])
prefixname = sys.argv[3]
skew_dist = int(sys.argv[4])
obs_mag_selection = int(sys.argv[5])


salt2_version = "salt3-22"
source = sncosmo.SALT3Source(modeldir = os.environ["PATHMODEL"] + "/" + salt2_version + "/")

nonSALTkeys = ["MB", "mass", "delta_m"]

SDSS_obs_frame = {}
for filt in "griz":
    SDSS_obs_frame[filt] = sncosmo.get_bandpass("sdss" + filt)

params = dict(salt2_version = salt2_version, n_visit = 200, ndeg2 = 5., nsnepernight = 3, ndataset = ndataset, cadence = 4.,
              obs_mag_selection = obs_mag_selection,
              Rx1 = 0.5 + 0.45*(1 - skew_dist), tau_x1 = -0.8*skew_dist,
              Rc = 0.05 + 0.035*(1 - skew_dist), tau_c = 0.07*skew_dist,
              gray_sig_unexplained = 0.12, alpha = 0.15,
              beta_B = 3.1, beta_R = 3.1, delta_beta_R = 0., delta = 0.08, delta_h = 0.5, MB = -19.1)

subprocess.getoutput("rm -fr " + prefixname)
subprocess.getoutput("mkdir " + prefixname)

f = open(prefixname + "/params.dat", 'w')
for param in params:
    f.write(param + "  " + str(params[param]) + '\n')
f.close()


f = open(prefixname + "/mag_cuts.txt", 'w')
for dataset_ind in range(ndataset):
    f.write("dataset_L_%03i_v1.txt  $UNITY/paramfiles/SDSS_r_selection.txt    18.0            0.5\n" % dataset_ind)
    f.write("dataset_H_%03i_v1.txt  $UNITY/paramfiles/SDSS_i_selection.txt    23.0            0.5\n" % dataset_ind)
f.close()

f = open(prefixname + "/mag_cuts_x0.txt", 'w')
for dataset_ind in range(ndataset):
    f.write("dataset_%03i_v1.txt  $UNITY/paramfiles/No_k_correct.txt    18.0            0.5\n" % dataset_ind)
    f.write("dataset_%03i_v1.txt  $UNITY/paramfiles/No_k_correct.txt    23.0            0.5\n" % dataset_ind)
f.close()


f = open(prefixname + "/calibration_uncertainties.txt", 'w')
f.write("""
('Fundamental', (3000.0, 4000.0)):                                                                      0.0001
('Fundamental', (4000.0, 5000.0)):                                                                      0.0001
('Fundamental', (6000.0, 8000.0)):                                                                      0.0001
('Fundamental', (8000.0, 100000.0)):                                                                    0.0001
('Fundamental', (10000.0, 100000.0)):                                                                   0.0001


"SALT_U_CAL":                                                                                           0.0001
"SALT_I_CAL":                                                                                           0.0001

('Zeropoint', 'SDSS|SDSS_u'): 0.005
('Zeropoint', 'SDSS|SDSS_g'): 0.005
('Zeropoint', 'SDSS|SDSS_r'): 0.005
('Zeropoint', 'SDSS|SDSS_i'): 0.005
('Zeropoint', 'SDSS|SDSS_z'): 0.005
('Lambda', 'SDSS|SDSS_u'):    0.01
('Lambda', 'SDSS|SDSS_g'):    0.01
('Lambda', 'SDSS|SDSS_r'):    0.01
('Lambda', 'SDSS|SDSS_i'):    0.01
('Lambda', 'SDSS|SDSS_z'):    0.01
""")
f.close()


f = open(prefixname + "/calibration_uncertainties_small.txt", 'w')
f.write("""
('Fundamental', (3000.0, 4000.0)):                                                                      0.0001
('Fundamental', (4000.0, 5000.0)):                                                                      0.0001
('Fundamental', (6000.0, 8000.0)):                                                                      0.0001
('Fundamental', (8000.0, 100000.0)):                                                                    0.0001
('Fundamental', (10000.0, 100000.0)):                                                                   0.0001


"SALT_U_CAL":                                                                                           0.0001
"SALT_I_CAL":                                                                                           0.0001

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
""")
f.close()



f = open(prefixname + "/weird_sn_list.txt", 'w')
f.close()


f_UNITY = open(prefixname + "/run_UNITY.sh", 'w')


f_UNITY.write("""#!/bin/bash
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

f_UNITY.write("cd " + pwd + "/" + prefixname + "\n")
f_UNITY.write("python $PATHMODEL/python_code/cut_fits.py dataset*\n")

for dataset_ind in tqdm.trange(ndataset):
    cal_offsets = {}
    for band in "griz":
        cal_offsets[band] = np.random.normal()*0.005*add_noise_and_calibration


    wd = prefixname + "/dataset_H_%03i/" % dataset_ind
    subprocess.getoutput("mkdir " + wd)
    make_dataset(wd, cal_offsets = cal_offsets)
    
    wd = prefixname + "/dataset_L_%03i/" % dataset_ind
    subprocess.getoutput("mkdir " + wd)
    make_dataset(wd, cal_offsets = cal_offsets)


    for include_low in [0, 1]:
        for oneDint, nocal, noselection, twopop in ([0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]):
            wd = prefixname + "/UNITY%s%s%s%s%s_%03i/" % ("L"*include_low + "H", "_1D"*oneDint,
                                                          "_nocal"*nocal, "_nosel"*noselection, "_twopop"*twopop, dataset_ind)
            subprocess.getoutput("mkdir " + wd)
            
            set_up_UNITY(wd, dataset_ind = dataset_ind, oneDint = oneDint, nocal = nocal, noselection = noselection, twopop = twopop, include_low = include_low)
            
            f_UNITY.write("cd " + pwd + "/" + wd + '\n')
            f_UNITY.write("sbatch run.sh\n")
f_UNITY.close()
