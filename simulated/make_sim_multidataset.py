import subprocess
import matplotlib.pyplot as plt
from numpy import *
from FileRead import readcol
from scipy.interpolate import interp1d
import os
import tqdm
import sys

UNITY = os.environ["UNITY"]

def do_it(cmd):
    print(cmd)
    print(subprocess.getoutput(cmd))



sim_sel_effects = int(sys.argv[1])
fit_sel_effects = int(sys.argv[2])
sim_sig_int = int(sys.argv[3])
sim_two_beta = int(sys.argv[4])
fit_two_beta = int(sys.argv[5])
k_correct = int(sys.argv[6])



nsne = [300, 150, 150]
if k_correct:
    sel_fl = ["SDSS_r_selection.txt", "SDSS_r_selection.txt", "SDSS_i_selection.txt"]
else:
    sel_fl = ["No_k_correct.txt"]*3
datasetnames = ["Nearby", "SDSS", "SNLS"]
if sim_sel_effects:
    mag_limits = [18.5, 21., 24.]
else:
    mag_limits = [21., 24., 27.]
max_z = [0.1, 0.5, 1.0]

dc = 0.03
dx1 = 0.3
dm = 0.03

if sim_sig_int:
    sig_unexpl = 0.1
else:
    sig_unexpl = 0.

beta_unexpl = 3.1 # For converting magnitude to color
if sim_two_beta:
    beta_B = 2.
    beta_R = 4.
else:
    beta_B = 3.1
    beta_R = 3.1

for sim_ind in tqdm.trange(0, 400):
    sim_wd = "simulated_data_runs/sim_%03i/" % sim_ind
    subprocess.getoutput("rm -fr " + sim_wd)
    subprocess.getoutput("mkdir -p " + sim_wd)


    frac_c_var = random.random()
    c_sig_int = sqrt(frac_c_var) * sig_unexpl / beta_unexpl
    m_sig_int = sqrt(1 - frac_c_var) * sig_unexpl

    print("frac_c_var, c_sig_int, m_sig_int", frac_c_var, c_sig_int, m_sig_int)

    f = open(sim_wd + "/mag_cuts.txt", 'w')
    f.write("#sample			kc_file						est_cut_value	est_cut_sigma\n")
    for samp_ind in range(len(nsne)):
        f.write(datasetnames[samp_ind] + "_sim.txt	        $UNITY/paramfiles/" + sel_fl[samp_ind] + "          " + str(mag_limits[samp_ind]) + "            0.5\n")
    f.close()

    f = open(sim_wd + "paramfile.txt", 'w')
    f.write("""do_blind                0

filenamelist            ["Nearby_sim.txt","SDSS_sim.txt","SNLS_sim.txt"]

weird_sn_list           "weird_sn_list.txt"
mag_cut                 "mag_cuts.txt"
    stan_code               '$UNITY/scripts/""" + "stan_code_simple_fixsel.txt"*(fit_sel_effects == 2) + "stan_code_simple.txt"*(fit_sel_effects == 1) + (fit_sel_effects == 0)*"stan_code_simple_no_sel.txt" + """'
sample_file             "None"


min_redshift            0.01
max_redshift            3.
max_firstphase          6.
min_lastphase           0.
max_color_uncertainty   0.2
max_color               0.25
max_MWEBV               0.25
min_color               -0.25
remap_x1                [0.,0.]

# Units of c:
pec_vel_disp            0.001
# Units of magnitudes per redshift
lensing_disp            0.055
MWEBV_zeropoint_EBV     0.005
outl_frac               0.02
calib_errs              
n_x1c_star              3
electron_coeff          [0.0039,0.001]
IG_extinction_coeff     1.


do_twoalphabeta         """ + str(fit_two_beta) + """
threeD_unexplained      1



iter                    2000
n_jobs                  4
chains                  4

do_host_mass            1
fix_Om                  0
MB_by_sample            0
include_pec_cov         0
separate_mass_x1c       1""")
    f.close()


    f = open(sim_wd + "weird_sn_list.txt", 'w')
    f.close()


    for samp_ind in range(len(nsne)):
        [rband_z, rband_c2, rband_c3] = readcol(UNITY + "/paramfiles/" + sel_fl[samp_ind], 'fff')

        ifnc2 = interp1d(rband_z, rband_c2, kind = 'linear')
        ifnc3 = interp1d(rband_z, rband_c3, kind = 'linear')

        redshifts = random.random(size = nsne[samp_ind]*10)**0.5 * max_z[samp_ind]


        rs = (1.00038*redshifts - 0.227753*redshifts**2 - 0.0440361*redshifts**3 + 0.0619502*redshifts**4 -  0.0220087*redshifts**5 + 0.00289242*redshifts**6)
        mus = 5*log10((1. + redshifts)*rs) + 43.1586133146


        masses = random.normal(size = nsne[samp_ind]*10) + 10.
        high_mass = array((masses > 10), dtype=float64)


        cBs = random.normal(size = nsne[samp_ind]*10)*0.05 - 0.08
        cRs = random.exponential(size = nsne[samp_ind]*10)*0.075
        
        x1s = random.normal(size = nsne[samp_ind]*10)*0.5 + 0.5 - random.exponential(size = nsne[samp_ind]*10)*(0.5 + high_mass)

        mBs = -19.1 + mus -0.14*x1s + 4.*cRs + 2.*cBs # - (masses > 10.)*0.08

        obs_cs = cBs + cRs + random.normal(size = nsne[samp_ind]*10)*dc + random.normal(size = nsne[samp_ind]*10)*c_sig_int
        obs_x1s = x1s + random.normal(size = nsne[samp_ind]*10)*dx1
        obs_mBs = mBs + random.normal(size = nsne[samp_ind]*10)*dm + random.normal(size = nsne[samp_ind]*10
        )*redshifts*0.055 + random.normal(size = nsne[samp_ind]*10)*0.0022/redshifts + random.normal(size = nsne[samp_ind]*10)*m_sig_int

        obs_rbands = obs_mBs + ifnc2(redshifts) + ifnc3(redshifts)*obs_cs

        plt.subplot(2,1,1)
        plt.scatter(redshifts, obs_mBs, c = obs_cs, s = 1)
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.hist(obs_mBs, bins = 20)
        plt.savefig("pre-selection_data.pdf")
        plt.close()

        inds = where(obs_rbands + random.normal(size = nsne[samp_ind]*10)*0.5 < mag_limits[samp_ind])
        redshifts, masses, obs_mBs, obs_x1s, obs_cs = [item[inds] for item in [redshifts, masses, obs_mBs, obs_x1s, obs_cs]]

        plt.subplot(2,1,1)
        plt.scatter(redshifts, obs_mBs, c = obs_cs, s = 1)
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.hist(obs_mBs, bins = 20)
        plt.savefig("post-selection_data.pdf")
        plt.close()

        plt.figure(figsize = (10, 20))
        plt.plot(redshifts, obs_cs, '.')
        zbins = linspace(0., max_z[samp_ind]*1.01, 10)
        for i in range(len(zbins) - 1):
            inds = where((redshifts > zbins[i])*(redshifts <= zbins[i+1]))
            plt.plot(mean(redshifts[inds]), mean(obs_cs[inds]), 'o', color = 'r')

        for y in arange(-0.1, 0.31, 0.1):
            plt.axhline(y)

        plt.xlabel("Redshift")
        plt.ylabel("Observed Color")
        plt.savefig("c_vs_z.pdf", bbox_inches = 'tight')
        plt.close()
        
        redshifts, masses, obs_mBs, obs_x1s, obs_cs = [item[:nsne[samp_ind]] for item in [redshifts, masses, obs_mBs, obs_x1s, obs_cs]]
        assert len(redshifts) == nsne[samp_ind]


        f1 = open(sim_wd + datasetnames[samp_ind] + "_sim.txt", 'w')

        for i in range(nsne[samp_ind]):
            SN_name = "SN_sim_%04i" % i
            wd = sim_wd + datasetnames[samp_ind] + "/" + SN_name
            do_it("mkdir -p " + wd)

            f1.write(subprocess.getoutput("pwd") + "/" + wd + '\n')

            f = open(wd + "/result_salt2.dat", 'w')
            f.write("Redshift  %f\n" % redshifts[i])
            f.write("FirstPhase  %f\n" % -10.)
            f.write("LastPhase  %f\n" % 30.)
            f.write("RestFrameMag_0_B  %f  %f\n" % (obs_mBs[i], dm))
            f.write("Color  %f  %f\n" % (obs_cs[i], dc))
            f.write("X1  %f  %f\n" % (obs_x1s[i], dx1))
            f.write("CovRestFrameMag_0_BX1  0.0\n")
            f.write("CovColorRestFrameMag_0_B  0.0\n")
            f.write("CovColorX1  0.0\n")
            f.write("CovX1X1  %f\n" % (dx1**2.))
            f.write("CovColorColor  %f\n" % (dc**2.))
            f.close()

            f = open(wd + "/lightfile", 'w')
            f.write("z_cmb  %f\n" % redshifts[i])
            f.write("z_heliocentric  %f\n" % redshifts[i])
            f.write("RA  0.0\n")
            f.write("DEC  0.0\n")
            f.write("MWEBV  0.0\n")
            f.write("Mass  %f  -0.05  0.05" % masses[i])
            f.close()

            f = open(wd + "/result_deriv.dat", 'w')
            f.write("""#Parameter      MagSys|Instrument|Band    RestLamb         Phase    dmu/dP                dmB/dP                ds/dP                 dc/dP                 
Zeropoint                              VEGA|SDSS|SDSS_g          3986.33668043    All      0.                     0.                   0.0                   0.         
Zeropoint                              VEGA|SDSS|SDSS_i          6316.75624541    All      0.                     0.                   0.0                   0.        
MWEBV                                  All|All|All               All              All      0.0                    0.0                  0.0                   0.0             
Check                                  All|All|All               All              All      1.0                    1.0                  0.0                   0.0    
""")
            f.close()

        f1.close()

    

