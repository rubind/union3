import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import subprocess
from scipy.stats import scoreatpercentile
from DavidsNM import miniLM_new
import glob
import tqdm
from astropy.cosmology import FlatLambdaCDM


def pullfn(P, passdata):
    [redshifts, obs_mus] = passdata[0]
    cosmo = FlatLambdaCDM(Om0 = P[1], H0 = 70.)
    mu = cosmo.distmod(redshifts).value

    return obs_mus - (P[0] + mu)
    

drs = glob.glob("UNITYLH_???/input*pickle")
drs = [item.split("/")[0] for item in drs]

all_freq_Oms = []
all_UNITY_Oms = []
all_drs = []
all_beta_B = []
all_sig_int = []
all_tau_c = []
all_c_star = []
all_mobs_cuts = []

for dr in tqdm.tqdm(drs):
    the_data, stan_data, params = pickle.load(gzip.open(dr + "/inputs__.pickle", "rb"))

    for key in the_data:
        print(key)

    obs_mus = the_data["mB_list"] + 0.15*the_data["x1_list"] - 3.1*the_data["c_list"]

    P, NA, NA = miniLM_new(ministart = [0., 0.], miniscale = [1., 1.], residfn = pullfn, passdata = [the_data["z_CMB_list"], obs_mus])

    print(P)

    """
    z_bins = scoreatpercentile(redshifts, np.linspace(0., 100., 11))
    z_bins[0] -= 0.001
    z_bins[1] += 0.001

    binned_HR = []
    binned_z = []

    for i in range(len(z_bins) - 1):
        inds = np.where((redshifts >= z_bins[i])*(redshifts < z_bins[i+1]))
        binned_HR.append(np.mean(HRs[inds]))
        binned_z.append(np.mean(redshifts[inds]))

    plt.scatter(binned_z, binned_HR)
    plt.axhline(0)
    plt.savefig("HR.pdf")
    plt.close()
    """

    try:
        this_Om = float(subprocess.getoutput("grep 'Om ' " + dr + "/log.txt").split(None)[1])
        all_UNITY_Oms.append(this_Om)
        all_freq_Oms.append(0.3 + P[1])
        all_drs.append(dr)
        all_beta_B.append(float(subprocess.getoutput("grep beta_B " + dr + "/log.txt").split(None)[1]))

        all_sig_int.append([float(item.split(None)[1]) for item in subprocess.getoutput("grep sigma_int " + dr + "/log.txt").split('\n')])
        all_tau_c.append([float(item.split(None)[1]) for item in subprocess.getoutput("grep 'tau_c\[' " + dr + "/log.txt").split('\n')])
        all_c_star.append([float(item.split(None)[1]) for item in subprocess.getoutput("grep 'c_star\[' " + dr + "/log.txt").split('\n')])
        all_mobs_cuts.append([float(item.split(None)[1]) for item in subprocess.getoutput("grep 'mobs_cuts\[' " + dr + "/log.txt").split('\n')])
    except:
        pass

all_UNITY_Oms = np.array(all_UNITY_Oms)
all_freq_Oms = np.array(all_freq_Oms)
all_sig_int = np.array(all_sig_int)
all_tau_c = np.array(all_tau_c)
all_c_star = np.array(all_c_star)
all_mobs_cuts = np.array(all_mobs_cuts)

plt.scatter(all_UNITY_Oms, all_freq_Oms, c = all_beta_B)

for i in range(len(all_freq_Oms)):
    if np.abs(all_UNITY_Oms[i] - all_freq_Oms[i]) > 0.03:
        plt.text(all_UNITY_Oms[i], all_freq_Oms[i], all_drs[i], fontsize = 8)
plt.axhline(np.mean(all_freq_Oms), color = 'k', label = "%.4f +- %.4f" % (np.mean(all_freq_Oms), np.std(all_freq_Oms)/np.sqrt(len(all_freq_Oms))))
plt.axvline(np.mean(all_UNITY_Oms), color = 'b', label = "%.4f +- %.4f" % (np.mean(all_UNITY_Oms), np.std(all_UNITY_Oms)/np.sqrt(len(all_UNITY_Oms))))
plt.title("all_freq_Oms %.3f all_UNITY_Oms %.3f average %.3f\ndiff %.4f +- %.4f" % (np.std(all_freq_Oms), np.std(all_UNITY_Oms), np.std(0.5*(all_freq_Oms + all_UNITY_Oms)),
                                                                                    np.mean(all_freq_Oms - all_UNITY_Oms), np.std(all_freq_Oms - all_UNITY_Oms)/np.sqrt(len(all_freq_Oms)) ))
plt.legend(loc = 'best')
plt.savefig("Om_vs_Om.pdf")
plt.close()

plt.hist(np.array(all_sig_int))
plt.savefig("sig_int.pdf")
plt.close()




for i in range(3):
    plt.scatter(all_sig_int[:,i], all_UNITY_Oms)
plt.savefig("Om_vs_sig_int.pdf")
plt.close()

for i in range(3):
    plt.scatter(all_tau_c[:,i], all_UNITY_Oms)
plt.savefig("Om_vs_tau_c.pdf")
plt.close()

for i in range(3):
    plt.scatter(all_c_star[:,i], all_UNITY_Oms)
plt.savefig("Om_vs_c_star.pdf")
plt.close()

for i in range(3):
    plt.subplot(1,3,1+i)
    plt.scatter(all_mobs_cuts[:,i], all_UNITY_Oms)
plt.savefig("Om_vs_mobs_cuts.pdf")
plt.close()
