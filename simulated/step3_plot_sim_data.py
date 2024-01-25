import glob
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from FileRead import read_param, readcol
from astropy.cosmology import FlatLambdaCDM
from DavidsNM import miniNM_new
import sys
import pickle
from scipy.special import erf

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
    

fig = plt.figure(figsize = (8, 6))

for lowhigh in "LHV":
    pltcolor = dict(L = 'b', H = 'g', V = 'r')[lowhigh]
    datalabel = dict(L = "Low-$z$", H = "Mid-$z$", V = "High-$z$")[lowhigh]


    all_data = read_or_load(lowhigh)
    
    if lowhigh == "H":
        z_step = 0.02
    elif lowhigh == "V":
        z_step = 0.2
    else:
        z_step = 0.002

    zbins = np.arange(all_data["all_z"].min(), all_data["all_z"].max() + z_step, z_step)
    magbins = np.arange(all_data["all_mags"].min(), all_data["all_mags"].max() + 0.1, 0.1)

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

    inds = np.where(all_data["all_incl"] == 1)

    plt.subplot(2,2,2)
    #plt.hist(all_z, bins = zbins, color = 'w', edgecolor='black', label = "All " + datalabel + " SNe")
    plt.hist(all_data["all_z"], bins = zbins, color = pltcolor, label = "All " + datalabel + " SNe", alpha = 0.3)
    plt.hist(all_data["all_z"][inds], bins = zbins, color = pltcolor, label = datalabel + " SNe Selected")
    plt.legend(loc = 'best')
    plt.ylabel("Number of SNe per Bin")
    plt.xlabel("Redshift")
    plt.xscale('log')
    plt.xlim(0.01, 3.5)

    if lowhigh == "H":

        z_step = 0.05
        zbins = np.arange(all_data["all_z"].min(), all_data["all_z"].max() + z_step, z_step)

        
        plt.subplot(2,2,3)
        
        inds = np.where(all_data["all_incl"] == 1)
        
        print("all_dm[inds]", all_data["all_dm"][inds])
        x,y, sigy=dobin(all_data["all_z"][inds], all_data["all_dm"][inds], zbins)
        plt.errorbar(x, y, yerr = sigy, fmt = '.', color = pltcolor)

        cosmo = FlatLambdaCDM(Om0 = 0.3, H0 = 70.)
        cosmo2 = FlatLambdaCDM(Om0 = 0.5, H0 = 70.)
        cosmo3 = FlatLambdaCDM(Om0 = 0.38, H0 = 70.)


        pltz = np.linspace(1e-5, 1, 100)

        mu = cosmo.distmod(z=pltz).value
        mu2 = cosmo2.distmod(z=pltz).value
        mu3 = cosmo3.distmod(z=pltz).value

        plt.axhline(0, color = 'k')

        #def chi2fn(P, passdata):
        #    dm = passdata[0]
        #    mu_minus_mu = passdata[1]

        #    return sum((dm + P[0] - mu_minus_mu)**2.)

        #P, NA, NA = miniNM_new(ministart = [0.], miniscale = [1.], chi2fn=chi2fn, passdata = (all_dm[inds], cosmo.distmod(z=all_z[inds]).value - cosmo2.distmod(z=all_z[inds]).value)

        inds = np.where(all_data["all_incl"] == 1)

        mean_resid = np.mean(all_data["all_dm"][inds] - (cosmo2.distmod(z=all_data["all_z"][inds]).value - cosmo.distmod(z=all_data["all_z"][inds]).value))
        plt.plot(pltz, mu2-mu + mean_resid, color = 'r', label = "$\mu(\Omega_m=0.5) - \mu(\Omega_m = 0.3)$")

        mean_resid = np.mean(all_data["all_dm"][inds] - (cosmo3.distmod(z=all_data["all_z"][inds]).value - cosmo.distmod(z=all_data["all_z"][inds]).value))
        plt.plot(pltz, mu3-mu + mean_resid, '--', color = 'g', label = "$\mu(\Omega_m=0.38) - \mu(\Omega_m = 0.3)$")


        plt.legend(loc = 'best')


    plt.subplot(2,2,4)
    
    
    print("all_dm[inds]", all_data["all_dm"][inds])
    x,y, sigy=dobin(all_data["all_z"][inds], all_data["all_dm"][inds], zbins)
    plt.errorbar(x, y, yerr = sigy, fmt = '.', color = pltcolor)



plt.xlabel("Redshift")
#plt.xscale('log')
#plt.xlim(0.01, 3.5)
plt.ylabel("Hubble Residual (Magnitudes)")

fig.align_ylabels()

plt.tight_layout()

plt.savefig("sim_data.pdf", bbox_inches = 'tight')
plt.close()

for lowhigh in "LHV":
    all_data = read_or_load(lowhigh)

    for key in ["found", "incl"]:
        P, NA, Cmat, = miniNM_new(ministart = [20., 5., 0.5, 0.0], miniscale = [2., 1., 0.1, 0.1], chi2fn = chi2fn_mag, passdata = [all_data["all_mags"], all_data["all_" + key]], verbose = False, compute_Cmat = False)
        print(lowhigh, key, P)

    
                            
