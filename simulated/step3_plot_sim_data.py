import glob
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from FileRead import read_param, readcol
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
from DavidsNM import miniNM_new
import sys
import pickle
from scipy.special import erf
from scipy.stats import scoreatpercentile

def chi2fn_mag(P, passdata):
    all_mags, all_incl = passdata[0]

    if P[1] < 0.01:
        return 1e100

    """
    if P[2]*2 + P[3] > 1:
        return 1e100

    if P[3] < 0:
        return 1e100

    if P[2] < 0:
        return 1e100
    """
    
    mod = P[2]*(1. + erf((P[0] - all_mags)/P[1])) + P[3]


    chi2 = all_incl - mod
    return np.dot(chi2, chi2)
    #chi2 = -((-1 + all_incl)*np.log(1 - mod)) + all_incl*np.log(mod)
    return sum(chi2)

    
def dobin(xs, ys, xbins):
    ybins = []
    sigybins = []
    
    for i in range(len(xbins)-1):
        inds = np.where((xs >= xbins[i])*(xs < xbins[i+1]))
        if len(inds[0]) > 4:
            ybins.append(np.mean(ys[inds]))
            sigybins.append(np.std(ys[inds], ddof=1)/np.sqrt(float(len(inds[0]))))
        else:
            ybins.append(np.sqrt(-1.))
            sigybins.append(np.sqrt(-1.))
            
    return 0.5*(xbins[1:] + xbins[:-1]), ybins, sigybins


def read_or_load(lowhigh):
    if sys.argv[1] == "read":
        all_data = {}
        
        all_data["all_mags"] = []
        all_data["all_found"] = []
        all_data["all_incl"] = []
        all_data["all_outl"] = []
        all_data["all_dm"] = []
        all_data["all_z"] = []

        globstr = "UNITY_" + lowhigh + "_???"

        all_v1_SNe = []
        for fl in glob.glob("dataset_" + lowhigh + "_???_v1.txt"):
            [v1_SNe] = readcol(fl, 'a')
            v1_SNe = [item.replace("$UNION/", "") for item in v1_SNe]
            all_v1_SNe += v1_SNe
            # E.g., dataset_H_099/SN0023

        for fl in tqdm.tqdm(glob.glob(globstr + "/SN_params/*")):
            # E.g., UNITY_H_000/SN_params/params_0000.dat
            all_data["all_mags"].append(read_param(fl, "peak_mag"))
            all_data["all_found"].append(read_param(fl, "observed"))
            all_data["all_outl"].append(read_param(fl, "outlier"))

            SN_path = fl.replace("UNITY", "dataset").replace("SN_params/params_", "SN").replace(".dat", "")

            all_data["all_incl"].append(all_v1_SNe.count(SN_path))
            print("incl", all_data["all_incl"][-1])
            all_data["all_dm"].append(read_param(fl, "delta_mu"))
            all_data["all_z"].append(read_param(fl, "z"))

        
        for key in all_data:
            all_data[key] = np.array(all_data[key])

                
        pickle.dump(all_data, open("sim_truth_" + lowhigh + ".pickle", 'wb'))
    elif sys.argv[1] == "load":
        all_data = pickle.load(open("sim_truth_" + lowhigh + ".pickle", 'rb'))

        
    return all_data

def get_cosmo_label(cos_default, cos_offset):
    if cos_offset.w0 == cos_default.w0:
        return "\mu(\Omega_m=%.2f) - \mu(\Omega_m=%.2f)" % (cos_offset.Om0, cos_default.Om0)
    else:
        return "\mu(\Omega_m=%.2f,\ w_0=%.2f,\ w_a=%.2f)$\n$ - \mu(\Omega_m=%.2f,\ w_0=%.2f,\ w_a=%.2f)" % (cos_offset.Om0, cos_offset.w0, cos_offset.wa,
                                                                                                          cos_default.Om0, cos_default.w0, cos_default.wa)

def show_other_cosmos(all_data, inds, cosmo, cosmo2, cosmo3):
    #z_step = 0.05
    #zbins = np.arange(all_data["all_z"].min(), all_data["all_z"].max() + z_step, z_step)




    #print("all_dm[inds]", all_data["all_dm"][inds])
    #x,y, sigy=dobin(all_data["all_z"][inds], all_data["all_dm"][inds], zbins)
    #plt.errorbar(x, y, yerr = sigy, fmt = '.', color = pltcolor)


    xlim = plt.xlim()
    
    pltz = np.linspace(1e-5, xlim[1], 200)

    mu = cosmo.distmod(z=pltz).value
    mu2 = cosmo2.distmod(z=pltz).value
    mu3 = cosmo3.distmod(z=pltz).value

    plt.axhline(0, color = 'k')

    
    mean_resid = np.mean(all_data["all_dm"][inds] - (cosmo2.distmod(z=all_data["all_z"][inds]).value - cosmo.distmod(z=all_data["all_z"][inds]).value))
    plt.plot(pltz, mu2-mu + mean_resid, color = 'orange', label = "$%s$" % get_cosmo_label(cosmo, cosmo2), linewidth = 2, zorder = 5)

    mean_resid = np.mean(all_data["all_dm"][inds] - (cosmo3.distmod(z=all_data["all_z"][inds]).value - cosmo.distmod(z=all_data["all_z"][inds]).value))
    plt.plot(pltz, mu3-mu + mean_resid, '--', color = 'purple', label = "$%s$" % get_cosmo_label(cosmo, cosmo3), linewidth = 2, zorder = 5)


    plt.legend(loc = 'best')

    

fig = plt.figure(figsize = (10, 8))

all_for_cosmo = {}

for lowhigh in "LHV":
    pltcolor = dict(L = 'b', H = 'g', V = 'r')[lowhigh]
    datalabel = dict(L = "Low-$z$", H = "Mid-$z$", V = "High-$z$")[lowhigh]


    all_data = read_or_load(lowhigh)
    
    if lowhigh == "H":
        z_step = 0.02
        n_mag_bins = 100
    elif lowhigh == "V":
        z_step = 0.2
        n_mag_bins = 30
    else:
        z_step = 0.002
        n_mag_bins = 100

    zbins = np.arange(all_data["all_z"].min(), all_data["all_z"].max() + z_step, z_step)
    magbins = scoreatpercentile(all_data["all_mags"], np.linspace(0, 100, n_mag_bins)) #np.arange(all_data["all_mags"].min(), all_data["all_mags"].max() + 0.2, 0.2)
    magbins[0] -= 0.001
    magbins[-1] += 0.001
    
    """
    plt.subplot(2,2,2)
    x,y, NA=dobin(all_z, all_incl, zbins)
    plt.plot(x, y, '.', color = pltcolor)
    plt.ylabel("Selection Probability")
    plt.xlabel("Redshift")
    #plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xscale('log')
    plt.xlim(0.01, 3.5)
    """
    
    plt.subplot(2,2,1)
    x,y,NA =dobin(all_data["all_mags"], all_data["all_incl"], magbins)
    plt.plot(x, y, '.', color = pltcolor)
    plt.ylabel("Selection Probability")
    plt.xlabel("Peak $r$, $i$, or $F125W$ Magnitude")
    plt.ylim(0,1)

    inds = np.where((all_data["all_incl"] == 1)*(all_data["all_outl"] == 0))
    for key in all_data:
        if key in all_for_cosmo:
            all_for_cosmo[key] = np.concatenate((all_for_cosmo[key], all_data[key][inds]))
        else:
            all_for_cosmo[key] = all_data[key][inds]

    plt.subplot(2,2,2)
    #plt.hist(all_z, bins = zbins, color = 'w', edgecolor='black', label = "All " + datalabel + " SNe")
    plt.hist(all_data["all_z"], bins = zbins, color = pltcolor, label = "All " + datalabel + " SNe Simulated", alpha = 0.3)
    plt.hist(all_data["all_z"][inds], bins = zbins, color = pltcolor, label = datalabel + " SNe Selected")
    plt.legend(loc = 'best')
    plt.ylabel("Number of Simulated SNe per Bin")
    plt.xlabel("Redshift")
    plt.xscale('log')
    plt.xlim(0.01, 3.0)

    if lowhigh == "H":
        plt.subplot(2,2,3)

        print("all_dm[inds]", all_data["all_dm"][inds])
        x,y, sigy=dobin(all_data["all_z"][inds], all_data["all_dm"][inds], zbins)
        plt.errorbar(x, y, yerr = sigy, fmt = '.', color = pltcolor)
        plt.ylabel("Mean Hubble Residual of Non-Outlier\nSelected Simulated SNe (Magnitudes)")

        
        cosmo = Flatw0waCDM(Om0 = 0.3, H0 = 70., w0 = -1, wa = 0)
        cosmo2 = Flatw0waCDM(Om0 = 0.52, H0 = 70., w0 = -1, wa = 0)
        cosmo3 = Flatw0waCDM(Om0 = 0.36, H0 = 70., w0 = -1, wa = 0)
        show_other_cosmos(all_data, inds, cosmo, cosmo2, cosmo3)
        plt.xlim(0, plt.xlim()[1])
        plt.xlabel("Redshift")

    plt.subplot(2,2,4)

    
    print("all_dm[inds]", all_data["all_dm"][inds])
    x,y, sigy=dobin(all_data["all_z"][inds], all_data["all_dm"][inds], zbins)
    plt.errorbar(x, y, yerr = sigy, fmt = '.', color = pltcolor)

plt.subplot(2,2,4)

cosmo = Flatw0waCDM(Om0 = 0.3, H0 = 70., w0 = -1, wa = 0)
cosmo2 = Flatw0waCDM(Om0 = 0.3, H0 = 70., w0 = -1.06, wa = 0.72)
cosmo3 = Flatw0waCDM(Om0 = 0.3, H0 = 70., w0 = -1.18, wa = 1.11)
show_other_cosmos(all_for_cosmo, inds = np.where(all_for_cosmo["all_mags"] > 0), cosmo = cosmo, cosmo2 = cosmo2, cosmo3 = cosmo3)
plt.xscale('log')
plt.xlim(0.01, 3)

plt.xlabel("Redshift")
#plt.xscale('log')
#plt.xlim(0.01, 3.5)
#plt.ylabel("Hubble Residual (Magnitudes)")
plt.ylabel("Mean Hubble Residual of Non-Outlier\nSelected Simulated SNe (Magnitudes)")

fig.align_xlabels()
fig.align_ylabels()

plt.tight_layout()

plt.figtext(0.98, 0.991, "Simulated Data", color = 'r', ha = 'right', va = 'top', bbox=dict(edgecolor = 'r', pad = 1, facecolor = 'w'))


plt.savefig("sim_data.pdf", bbox_inches = 'tight')
plt.close()

for lowhigh in "LHV":
    all_data = read_or_load(lowhigh)

    for key in ["found", "incl"]:
        P, NA, Cmat, = miniNM_new(ministart = [20., 5., 0.5, 0.0], miniscale = [2., 1., 0.1, 0.1], chi2fn = chi2fn_mag, passdata = [all_data["all_mags"], all_data["all_" + key]], verbose = False, compute_Cmat = False)
        print(lowhigh, key, P)

    
                            
