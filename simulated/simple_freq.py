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
import sys

def dobin(xs, ys, wys, meanbin):
    #xbins = 10.**np.linspace(np.log10(min(xs)*0.999),
    #                         np.log10(max(xs)*1.001), 21)

    xbins = scoreatpercentile(xs, np.linspace(0, 100, 21))
    xbins[0] *= 0.9999
    xbins[-1] *= 1.0001
    
    binx = []
    biny = []
    
    for i in range(len(xbins) - 1):
        inds = np.where((xs > xbins[i])*(xs <= xbins[i+1]))

        if meanbin:
            binx.append(sum(xs[inds]*wys[inds])/ sum(wys[inds]))
            biny.append(sum(ys[inds]*wys[inds])/ sum(wys[inds]))
        else:
            binx.append(np.median(xs[inds]))
            biny.append(np.median(ys[inds]))

    return binx, biny
                    
    

def pullfn(P, passdata):
    [redshifts, obs_mus, obs_dmus] = passdata[0]
    cosmo = FlatLambdaCDM(Om0 = P[1], H0 = 70.)
    mu = cosmo.distmod(redshifts).value

    return (obs_mus - (P[0] + mu))/obs_dmus


def make_bin_plot(xs, ys, sigys, binname):
    plt.figure(figsize = (8, 6))

    binx, biny = dobin(xs, ys, 1./sigys**2., meanbin = 1)
    plt.plot(binx, biny, '.', color = 'r', label = "Weighted Mean")

    binx, biny = dobin(xs, ys, np.ones(len(ys), dtype=np.float64), meanbin = 1)
    plt.plot(binx, biny, '.', color = 'g', label = "Mean")

    binx, biny = dobin(xs, ys, None, meanbin = 0)
    plt.plot(binx, biny, '.', color = 'b', label = "Median")

    plt.xscale('log')

    medval = np.median(ys)
    plt.axhline(medval, label = "$\Omega_m = 0.3$")

    tmpcosmo30 = FlatLambdaCDM(Om0 = 0.3, H0 = 70.)
    tmpcosmo29 = FlatLambdaCDM(Om0 = 0.29, H0 = 70.)

    xlim = plt.xlim()

    pltz = np.linspace(xlim[0], xlim[1], 100)
    pltHR = tmpcosmo29.distmod(pltz).value - tmpcosmo30.distmod(pltz).value

    plt.plot(pltz, pltHR + medval, label = "$\Omega_m = 0.29$")

    plt.legend(loc = 'best')
    plt.xlim(xlim)

    plt.savefig("bin_" + binname + ".pdf", bbox_inches = 'tight')
    plt.close()


prefix = sys.argv[1]

drs = glob.glob(prefix + "_???/input*pickle")
print("drs", drs)
drs = [item.split("/")[0] for item in drs]
print("drs", drs)

onealphabeta = [1, 0.15, -3.1]


all_freq_Oms = []
all_UNITY_Oms = []
all_drs = []
all_beta_B = []
all_sig_int = []
all_tau_c = []
all_c_star = []
all_mobs_cuts = []

all_zs = []
all_HRs = []
all_uncs = []
all_ax1s = [] # alpha x1
all_bcs = [] # beta c

for dr in tqdm.tqdm(drs):
    possible_inputs = glob.glob(dr + "/inputs*.pickle")
    assert len(possible_inputs) == 1
    
    the_data, stan_data, params = pickle.load(gzip.open(possible_inputs[0], "rb"))

    for key in the_data:
        print(key)

    for key in stan_data:
        print("stan_data", key)

    obs_mus = the_data["mB_list"] + onealphabeta[1]*the_data["x1_list"] + onealphabeta[2]*the_data["c_list"]

    tmpcosmo = FlatLambdaCDM(Om0 = 0.3, H0 = 70.)
    obs_HRs = obs_mus - tmpcosmo.distmod(the_data["z_CMB_list"]).value
    obs_dHRs = np.array([np.sqrt(np.dot(onealphabeta, np.dot(item, onealphabeta))) for item in stan_data["obs_mBx1c_cov"]])
    obs_dHRs = np.sqrt(   obs_dHRs**2. + 0.15**2. + (0.0022/the_data["z_CMB_list"])**2.   )
    
    all_zs.extend(the_data["z_CMB_list"])
    all_HRs.extend(obs_HRs)
    all_uncs.extend(obs_dHRs)
    all_ax1s.extend(onealphabeta[1]*np.array(the_data["x1_list"]))
    all_bcs.extend(onealphabeta[2]*np.array(the_data["c_list"]))
    
    P, NA, NA = miniLM_new(ministart = [0., 0.], miniscale = [1., 1.], residfn = pullfn, passdata = [the_data["z_CMB_list"], obs_mus, obs_dHRs])

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
        all_freq_Oms.append(P[1])
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
plt.savefig("Om_vs_Om_" + prefix + ".pdf")
plt.close()


all_zs = np.array(all_zs)
all_HRs = np.array(all_HRs)
all_uncs = np.array(all_uncs)
all_ax1s = np.array(all_ax1s)
all_bcs = np.array(all_bcs)

print("all_uncs", np.median(all_uncs))

make_bin_plot(xs = all_zs, ys = all_HRs, sigys = all_uncs, binname = "HR_from_03")
make_bin_plot(xs = all_zs, ys = all_ax1s, sigys = all_uncs, binname = "alpha_x1")
make_bin_plot(xs = all_zs, ys = all_bcs, sigys = all_uncs, binname = "beta_c")

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
