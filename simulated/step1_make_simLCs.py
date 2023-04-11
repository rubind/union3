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



def make_dataset(wd):

    obs_err = 5. 

    dates = np.arange(params["n_visit"], dtype=np.float64)*params["cadence"]


    source = sncosmo.SALT2Source(modeldir=os.environ["PATHMODEL"] + "/salt2-4/")
    model = sncosmo.Model(source=source)

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

        mabs = params["MB"] + p["delta_m"] - params["alpha"]*p["x1"] + 3.1*p["c"] - params["delta"]*(p["mass"] > 10.)*1.9*params["delta_h"]/(1. + 0.9*10.**(0.95*p["z"]))
        p["MB"] = mabs

        all_SNe.append(p)



    observed_SNe = np.zeros(nsne, dtype=np.int16)
        
    for night in dates:
        all_mags = []
        
        for i in range(nsne):
            if observed_SNe[i] == 0 and np.abs(night - all_SNe[i]["t0"]) < 50:
                model = get_SNCosmo_model(all_SNe[i], source)
                all_mags.append(model.bandmag("sdssi", "ab", night))
            else:
                all_mags.append(100.)
                
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
    for i in range(nsne):
        model = get_SNCosmo_model(all_SNe[i], source)
        peak_mags.append(model.bandmag("sdssi", "ab", all_SNe[i]["t0"]))
    
    peak_mags = np.array(peak_mags)
    plt.hist(peak_mags[np.where(observed_SNe)], alpha = 0.5)
    plt.hist(peak_mags[np.where(1 - observed_SNe)], alpha = 0.5)
    plt.savefig("selected_mags.pdf")
    plt.close()
    

    print("total observed_SNe", sum(observed_SNe))

    p_wd = wd + "/SN_params/"
    subprocess.getoutput("mkdir -p " + p_wd)

    cal_offsets = {}
    for band in "griz":
        cal_offsets[band] = np.random.normal()*0.005
    
    for i in range(nsne):

        f = open(p_wd + "/params_%04i.dat" % i, 'w')
        for key in all_SNe[i]:
            f.write(key + "  " + str(all_SNe[i][key]) + '\n')
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

            for band in ['sdssg', 'sdssr', 'sdssi', 'sdssz']:
                try:
                    fluxes, fluxcov = model.bandfluxcov(band, dates, zp = 27.5, zpsys = "ab")
                    good_band = 1
                except:
                    print("Couldn't get band", all_SNe[i]["z"], band)
                    good_band = 0

                    
                #print(fluxes)
                #print(fluxcov)
                if good_band:
                    for j in range(len(fluxcov)):
                        fluxcov[j,j] = np.clip(fluxcov[j,j], 0., fluxes[j]**2.)

                    model_fluxes = np.random.multivariate_normal(mean = fluxes, cov = fluxcov)
                    obs_fluxes = model_fluxes + np.random.normal(size = len(model_fluxes))*obs_err
                    
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
                        if np.abs(dates[j] - all_SNe[i]["t0"]) < 100:
                            towrite = [dates[j], obs_fluxes[j], obs_err, 27.5 + cal_offsets[band[-1]]]
                            towrite = [str(item) for item in towrite]
                            f.write("  ".join(towrite) + '\n')
                    f.close()
            

def set_up_UNITY(wd, dataset_ind):
    f = open(wd + "paramfile.txt", 'w')
    f.write("""
do_blind		0
filenamelist		["../dataset_%03i_v1.txt]

weird_sn_list		"../weird_sn_list.txt"
mag_cut			"../mag_cuts.txt"
stan_code		"$UNITY/scripts/stan_code_simple.txt"
sample_file		"None"
calibration_uncertainties		"../calibration_uncertainties.txt"


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
MWEBV_zeropoint_EBV	0.005
outl_frac		0.02
n_x1c_star		1
electron_coeff		[0.0042,0.00042]
IG_extinction_coeff	0.01


do_twoalphabeta		0
threeD_unexplained	1


iter			2500
n_jobs			4
chains			4

do_host_mass		1
fix_Om			0
MB_by_sample		0
include_pec_cov		0
separate_mass_x1c	1
    """ % (dataset_ind))
    f.close()


ndataset = int(sys.argv[1])

salt2_version = "salt3-22"
source = sncosmo.SALT3Source(modeldir = os.environ["PATHMODEL"] + "/" + salt2_version + "/")

nonSALTkeys = ["MB", "mass", "delta_m"]

params = dict(salt2_version = salt2_version, n_visit = 200, ndeg2 = 5., nsnepernight = 3, ndataset = ndataset, cadence = 4.,
              Rx1 = 0.5, tau_x1 = -0.8, Rc = 0.05, tau_c = 0.07,
              gray_sig_unexplained = 0.12, alpha = 0.15,
              beta_B = 3.1, beta_R = 3.1, delta_beta_R = 0., delta = 0.08, delta_h = 0.5, MB = -19.1)

subprocess.getoutput("rm -fr simLCs")
subprocess.getoutput("mkdir simLCs")

f = open("simLCs/params.dat", 'w')
for param in params:
    f.write(param + "  " + str(params[param]) + '\n')
f.close()


f = open("simLCs/mag_limits.txt", 'w')
for dataset_ind in range(ndataset):
    f.write("dataset_%03i_v1.txt  $UNITY/paramfiles/MEGACAMJLA_i_selection.txt    23.0            0.5\n" % dataset_ind)
f.close()

f = open("simLCs/calibration_uncertainties.txt", 'w')
f.write("""
('Fundamental', (3000.0, 4000.0)):                                                                      0.0001
('Fundamental', (4000.0, 5000.0)):                                                                      0.0001
('Fundamental', (6000.0, 8000.0)):                                                                      0.0001
('Fundamental', (8000.0, 100000.0)):                                                                    0.0001
('Fundamental', (10000.0, 100000.0)):                                                                   0.0001


"SALT_U_CAL":                                                                                           0.0001
"SALT_I_CAL":                                                                                           0.0001

('Zeropoint', 'SDSS|SDSS_u'):0.01
('Zeropoint', 'SDSS|SDSS_g'): 0.005
('Zeropoint', 'SDSS|SDSS_r'): 0.005
('Zeropoint', 'SDSS|SDSS_i'): 0.005
('Zeropoint', 'SDSS|SDSS_z'): 0.005
('Lambda', 'SDSS|SDSS_u'):    0.1
('Lambda', 'SDSS|SDSS_g'):    0.1
('Lambda', 'SDSS|SDSS_r'):    0.1
('Lambda', 'SDSS|SDSS_i'):    0.1
('Lambda', 'SDSS|SDSS_z'):    0.1
""")
f.close()

f = open("simLCs/weird_sn_list.txt", 'w')
f.close()


for dataset_ind in tqdm.trange(ndataset):
    wd = "simLCs/dataset_%03i/" % dataset_ind
    subprocess.getoutput("mkdir " + wd)

    make_dataset(wd)

    
    wd = "simLCs/UNITY_%03i/" % dataset_ind
    subprocess.getoutput("mkdir " + wd)

    f = open("simLCs/mag_limits.txt", 'a')
    f.write("dataset_%03i_v1.txt  \n")
    f.close()
    
    
    """set_up_UNITY(wd) make sure unblind is true"""
