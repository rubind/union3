import glob
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from FileRead import read_param
from astropy.cosmology import FlatLambdaCDM
from DavidsNM import miniNM_new
import sys
import pickle

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


fig = plt.figure(figsize = (8, 6))

for lowhigh in "LHV":
    pltcolor = dict(L = 'b', H = 'g', V = 'r')[lowhigh]
    datalabel = dict(L = "Low-$z$", H = "Mid-$z$", V = "High-$z$")[lowhigh]

    if sys.argv[1] == "read":
        all_mags = []
        all_obs = []
        all_dm = []
        all_z = []

        globstr = "UNITY_" + lowhigh + "_???"

        for fl in tqdm.tqdm(glob.glob(globstr + "/SN_params/*")):
            all_mags.append(read_param(fl, "peak_mag"))
            all_obs.append(read_param(fl, "observed"))
            all_dm.append(read_param(fl, "delta_mu"))
            all_z.append(read_param(fl, "z"))

        all_mags = np.array(all_mags)
        all_obs = np.array(all_obs)
        all_dm = np.array(all_dm)
        all_z = np.array(all_z)

        pickle.dump([all_mags, all_obs, all_dm, all_z], open("sim_truth_" + lowhigh + ".pickle", 'wb'))
    elif sys.argv[1] == "load":
        [all_mags, all_obs, all_dm, all_z] = pickle.load(open("sim_truth_" + lowhigh + ".pickle", 'rb'))

    if lowhigh == "H":
        z_step = 0.02
    elif lowhigh == "V":
        z_step = 0.2
    else:
        z_step = 0.002

    zbins = np.arange(all_z.min(), all_z.max() + z_step, z_step)
    magbins = np.arange(all_mags.min(), all_mags.max() + 0.1, 0.1)

    """
    plt.subplot(2,2,2)
    x,y, NA=dobin(all_z, all_obs, zbins)
    plt.plot(x, y, '.', color = pltcolor)
    plt.ylabel("Selection Probability")
    plt.xlabel("Redshift")
    #plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xscale('log')
    plt.xlim(0.01, 3.5)
    """
    
    plt.subplot(2,2,1)
    x,y,NA =dobin(all_mags, all_obs, magbins)
    plt.plot(x, y, '.', color = pltcolor)
    plt.ylabel("Selection Probability")
    plt.xlabel("Peak $r$, $i$, or $F125W$ Magnitude")
    plt.ylim(0,1)

    inds = np.where(all_obs == 1)

    plt.subplot(2,2,2)
    #plt.hist(all_z, bins = zbins, color = 'w', edgecolor='black', label = "All " + datalabel + " SNe")
    plt.hist(all_z, bins = zbins, color = pltcolor, label = "All " + datalabel + " SNe", alpha = 0.3)
    plt.hist(all_z[inds], bins = zbins, color = pltcolor, label = datalabel + " SNe Selected")
    plt.legend(loc = 'best')
    plt.ylabel("Number of SNe per Bin")
    plt.xlabel("Redshift")
    plt.xscale('log')
    plt.xlim(0.01, 3.5)

    if lowhigh == "H":

        z_step = 0.05
        zbins = np.arange(all_z.min(), all_z.max() + z_step, z_step)

        
        plt.subplot(2,2,3)
        
        inds = np.where(all_obs == 1)
        
        print("all_dm[inds]", all_dm[inds])
        x,y, sigy=dobin(all_z[inds], all_dm[inds], zbins)
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

        inds = np.where(all_obs == 1)

        mean_resid = np.mean(all_dm[inds] - (cosmo2.distmod(z=all_z[inds]).value - cosmo.distmod(z=all_z[inds]).value))
        plt.plot(pltz, mu2-mu + mean_resid, color = 'r', label = "$\mu(\Omega_m=0.5) - \mu(\Omega_m = 0.3)$")

        mean_resid = np.mean(all_dm[inds] - (cosmo3.distmod(z=all_z[inds]).value - cosmo.distmod(z=all_z[inds]).value))
        plt.plot(pltz, mu3-mu + mean_resid, '--', color = 'g', label = "$\mu(\Omega_m=0.38) - \mu(\Omega_m = 0.3)$")


        plt.legend(loc = 'best')


    plt.subplot(2,2,4)
    
    
    print("all_dm[inds]", all_dm[inds])
    x,y, sigy=dobin(all_z[inds], all_dm[inds], zbins)
    plt.errorbar(x, y, yerr = sigy, fmt = '.', color = pltcolor)



plt.xlabel("Redshift")
#plt.xscale('log')
#plt.xlim(0.01, 3.5)
plt.ylabel("Hubble Residual (Magnitudes)")

fig.align_ylabels()

plt.tight_layout()

plt.savefig("sim_data.pdf", bbox_inches = 'tight')
plt.close()

