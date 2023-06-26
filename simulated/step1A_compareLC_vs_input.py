from FileRead import readcol, read_param
import glob
import matplotlib.pyplot as plt
import numpy as np
from subprocess import getoutput
import tqdm
import sys
from scipy.stats import scoreatpercentile
from astropy.cosmology import FlatLambdaCDM
from DavidsNM import miniLM_new
import pickle

def do_med_bins(x, y, sigy, nbins, average_not_median = 0):
    bad_mask = np.isnan(x) + np.isnan(y) + np.isnan(sigy)
    
    inds = np.where(bad_mask == 0)
    binedges = scoreatpercentile(x[inds], np.linspace(0, 100, nbins+1))
    binedges[0] -= (binedges[1] - binedges[0])*0.001
    binedges[-1] += (binedges[-1] - binedges[-2])*0.001

    binx = []
    biny = []
    
    for i in range(nbins):
        inds = np.where((x >= binedges[i])*(x < binedges[i+1])*(bad_mask == 0))

        if average_not_median:
            if sum(1./sigy[inds]**2.) > 0:
                binx.append(sum(x[inds]/sigy[inds]**2.)/sum(1./sigy[inds]**2.))
                biny.append(sum(y[inds]/sigy[inds]**2.)/sum(1./sigy[inds]**2.))
            else:
                binx.append(np.sqrt(-1.))
                biny.append(np.sqrt(-1.))
        else:
            binx.append(np.median(x[inds]))
            biny.append(np.median(y[inds]))
    return binx, biny

def get_dmu(bands, resfl):
    f = open(resfl.replace("result_salt2.dat", "result_deriv.dat"), 'r')
    grepout = f.read().split('\n')
    f.close()
    
    grepout = [line for line in grepout if line.count("Zeropoint") == 1]
    

    dmudzps = dict(g = 0, r = 0, i = 0, z = 0)
    
    for band in bands:
        for line in grepout:
            if line.count("SDSS_" + band):
                dmudzps[band] = float(line.split(None)[4])
    return dmudzps


def modelfn(P, passdata):
    [zs, delta_mus] = passdata[0]

    cosmo = FlatLambdaCDM(Om0 = P[1], H0 = 70.)
    cosmo3 = FlatLambdaCDM(Om0 = 0.3, H0 = 70.)
    mu = cosmo.distmod(zs).value
    mu3 = cosmo3.distmod(zs).value
    return mu - mu3 + P[0]
    

def pullfn(P, passdata):
    [zs, delta_mus] = passdata[0]
    delta_mu_model = modelfn(P, passdata)
    
    return delta_mus - delta_mu_model


def fit_delta_cosmo(zs, delta_mus, pltzs):
    P, NA, NA = miniLM_new(ministart = [0.3, 0.0], miniscale = [1., 1.], residfn = pullfn, passdata = [zs, delta_mus])

    return modelfn(P, [[pltzs, None]]), "%.4f" % P[1]



    
all_dat = dict(true_c = [], delta_c = [], obs_sig_c = [],
               true_x1 = [], delta_x1 = [], obs_sig_x1 = [],
               delta_mag = [],
               delta_mu = [],
               obs_sig_mu = [],
               dmudg = [],
               dmudr = [],
               dmudi = [],
               dmudz = [],
               LH = [],
               redshift = [],
               resfl = [])

"""
if len(sys.argv) > 1:
    globstr = "dataset_*/*00*/result_deriv.dat"
else:
    globstr = "dataset_*/*/result_deriv.dat"

    
for resfl in tqdm.tqdm(glob.glob(globstr)):
    resfl = resfl.replace("result_deriv.dat", "result_salt2.dat")

    obs_c = read_param(resfl, "Color")
    if obs_c != None:
        obs_sig_c = read_param(resfl, "Color", ind = 2)


        obs_x0 = read_param(resfl, "X0")
        obs_mag = -2.5*np.log10(obs_x0)
        obs_x1 = read_param(resfl, "X1")
        obs_sig_x1 = read_param(resfl, "X1", ind = 2)

        
        redshift = read_param(resfl, "Redshift")

        paramsfl = resfl.replace("/SN", "/SN_params/params_").replace("/result_salt2.dat", ".dat").replace("dataset_", "UNITY_")

        
        true_x0 = read_param(paramsfl, "x0")
        true_mag = -2.5*np.log10(true_x0)
        
        true_x1 = read_param(paramsfl, "x1")
        true_c = read_param(paramsfl, "c")


        all_dat["resfl"].append(resfl)

        all_dat["true_c"].append(true_c)
        all_dat["delta_c"].append(obs_c - true_c)
        all_dat["obs_sig_c"].append(obs_sig_c)

        if resfl.count("_L_"):
            LH = "L"
        elif resfl.count("_H_"):
            LH = "H"
        else:
            assert 0, resfl
        
        all_dat["LH"].append(LH)

        all_dat["delta_mag"].append(obs_mag - true_mag)

        all_dat["true_x1"].append(true_x1)
        all_dat["delta_x1"].append(obs_x1 - true_x1)
        all_dat["obs_sig_x1"].append(obs_sig_x1)


        all_dat["delta_mu"].append((obs_mag + 0.14*obs_x1 - 3.1*obs_c) - (true_mag + 0.14*true_x1 - 3.1*true_c))
        all_dat["obs_sig_mu"].append(read_param(resfl, "dmu_estimate"))

        
        all_dat["redshift"].append(redshift)

        dmudzps = get_dmu("griz", resfl)
        for band in "griz":
            all_dat["dmud" + band].append(dmudzps[band])

for key in all_dat:
    all_dat[key] = np.array(all_dat[key])

all_dat["pulls_c"] = all_dat["delta_c"]/all_dat["obs_sig_c"]
all_dat["pulls_x1"] = all_dat["delta_x1"]/all_dat["obs_sig_x1"]


pickle.dump(all_dat, open("all_dat.pickle", 'wb'))
"""

all_dat = pickle.load(open("all_dat.pickle", 'rb'))

plt.figure(figsize = (36, 32))
for i, keys in enumerate([("redshift", "delta_mag", 0),
                          ("redshift", "delta_c", 0),
                          ("true_c", "delta_c", 0),
                          ("redshift", "delta_x1", 0),
                          ("redshift", "pulls_c", 0),
                          ("redshift", "pulls_x1", 0),
                          ("redshift", "delta_mu", 0),
                          ("redshift", "delta_mag", 35),
                          ("redshift", "delta_c", 35),
                          ("true_c", "delta_mu", 35),
                          ("true_x1", "delta_mu", 35),
                          ("true_c", "delta_c", 35),
                          ("redshift", "delta_x1", 35),
                          ("redshift", "delta_mu", 35),
                          ("redshift", "dmudg", 0),
                          ("redshift", "dmudg", 35),
                          ("dmudg", "delta_mag", 35),
                          ("dmudg", "delta_x1", 35),
                          ("dmudg", "delta_c", 35),
                          ("dmudg", "delta_mu", 35),
                          ("dmudr", "delta_mu", 35),
                          ("dmudi", "delta_mu", 35),
                          ("dmudz", "delta_mu", 35),
                          ("obs_sig_mu", "delta_mu", 35),
                          ("obs_sig_c", "delta_mu", 35),
                          ("obs_sig_c", "delta_c", 35),
                          ("true_x1", "delta_x1", 35)]):
    
    plt.subplot(6,5,i+1)
    if keys[2] == 0:
        plt.plot(all_dat[keys[0]], all_dat[keys[1]], '.', label = "Mean %.3f +- %.3f Median %.3f RMS %.3f" % (np.mean(all_dat[keys[1]]), np.std(all_dat[keys[1]], ddof=1)/np.sqrt(float(len(all_dat[keys[1]]))),
                                                                                                              np.median(all_dat[keys[1]]),
                                                                                                              np.std(all_dat[keys[1]], ddof=1)), color = 'b', alpha = 0.05)#, gridsize=100)
        
    else:

        for LH in "LH":
            pltcolor = dict(L = 'b', H = 'r')[LH]

            if keys[0] != "redshift":
                inds = np.where(all_dat["LH"] == LH)
            else:
                inds = np.where(all_dat["redshift"] > -1)
                
            nsne = len(all_dat[keys[0]][inds])
            binx, biny = do_med_bins(all_dat[keys[0]][inds], all_dat[keys[1]][inds], np.ones(nsne, dtype=np.float64), keys[2])
            plt.plot(binx, biny, '.', color = pltcolor, label = "Median")
           
            binx, biny = do_med_bins(all_dat[keys[0]][inds], all_dat[keys[1]][inds], np.ones(nsne, dtype=np.float64), keys[2], average_not_median = 1)
            plt.plot(binx, biny, '^', color = pltcolor, label = "Average")
            
            try:
                all_dat["obs_sig_" + keys[1].split("_")[-1]]
                has_uncs = 1
            except:
                has_uncs = 0
                
            if has_uncs:
                binx, biny = do_med_bins(all_dat[keys[0]][inds], all_dat[keys[1]][inds], all_dat["obs_sig_" + keys[1].split("_")[-1]][inds], keys[2], average_not_median = 1)
                plt.plot(binx, biny, '*', color = pltcolor, label = "Weighted")
                
                xlim = plt.xlim()
                pltx = np.linspace(0.01, xlim[1], 200)
                
                if keys[0] == "redshift" and keys[1] == "delta_mu":
                    plty, label = fit_delta_cosmo(binx, biny, pltx)
                    plt.plot(pltx, plty, label = label)
                

        plt.legend(loc = 'best')

        
    if keys[0] == "redshift":
        plt.xscale('log')
        plt.xlim(0.01, 1)

    plt.axhline(0)
    
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])

            
plt.tight_layout()
plt.savefig("compare_LC_vs_input.pdf", bbox_inches = 'tight')
plt.close()

