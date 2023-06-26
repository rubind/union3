from FileRead import readcol, read_param
import glob
import matplotlib.pyplot as plt
import numpy as np
from subprocess import getoutput
import tqdm
import sys
from scipy.stats import scoreatpercentile
from astropy.cosmology import FlatLambdaCDM
from DavidsNM import miniNM_new
import pickle

def do_med_bins(x, y, sigy, nbins, average_not_median = 0):
    binedges = scoreatpercentile(x, np.linspace(0, 100, nbins+1))
    binedges[0] -= (binedges[1] - binedges[0])*0.001
    binedges[-1] += (binedges[-1] - binedges[-2])*0.001

    binx = []
    biny = []
    
    for i in range(nbins):
        inds = np.where((x >= binedges[i])*(x < binedges[i+1]))

        if average_not_median:
            binx.append(sum(x[inds]/sigy[inds]**2.)/sum(1./sigy[inds]**2.))
            biny.append(sum(y[inds]/sigy[inds]**2.)/sum(1./sigy[inds]**2.))
        else:
            binx.append(np.median(x[inds]))
            biny.append(np.median(y[inds]))
    return np.array(binx), np.array(biny)


def modelfn(P, passdata):
    [zs, delta_mus, dmudg, dmudr, dmudi, dmudz] = passdata[0]

    cosmo = FlatLambdaCDM(Om0 = P[1], H0 = 70.)
    cosmo3 = FlatLambdaCDM(Om0 = 0.3, H0 = 70.)
    mu = cosmo.distmod(zs).value
    mu3 = cosmo3.distmod(zs).value
    return mu - mu3 + P[0] + P[2]*dmudg + P[3]*dmudr + P[4]*dmudi + P[5]*dmudz
    

def chifn(P, passdata): # Not chi2
    if P[1] <= 0:
        return 1e100
    
    [zs, delta_mus, dmudg, dmudr, dmudi, dmudz] = passdata[0]
    delta_mu_model = modelfn(P, passdata)
    
    resid = delta_mus - delta_mu_model
    return np.sum(np.abs(resid)) + np.sum((P[2:]/0.1)**2.)


def fit_delta_cosmo(zs, delta_mus, pltzs, dmudg, dmudr, dmudi, dmudz):
    passdata_orig = [zs, delta_mus, dmudg, dmudr, dmudi, dmudz]

    P, NA, NA = miniNM_new(ministart = [0.35, 0.0, 0.0, 0.0, 0.0, 0.0],
                           miniscale = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                           chi2fn = chifn, passdata = passdata_orig, verbose = True, compute_Cmat = False)


    print("P", P)

    
    P, NA, NA = miniNM_new(ministart = [0.35, 0.0, 0.0, 0.0, 0.0, 0.0],
                           miniscale = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
                           chi2fn = chifn, passdata = passdata_orig, verbose = True, compute_Cmat = False)

    print("P", P)
    
    med_bins, NA = do_med_bins(zs, delta_mus, np.ones(len(zs), dtype=np.float64), nbins = 200)
    
    passdata_binned = []
    for item in passdata_orig:
        tmp_med_bins, tmp_med_vals = do_med_bins(zs, item, np.ones(len(zs), dtype=np.float64), nbins = 200)

        assert np.all(med_bins == tmp_med_bins)
        passdata_binned.append(tmp_med_vals)

    
    P, NA, NA = miniNM_new(ministart = [0.35, 0.0, 0.0, 0.0, 0.0, 0.0],
                           miniscale = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
                           chi2fn = chifn, passdata = passdata_binned, verbose = True, compute_Cmat = False)

    print("P", P)
    

all_dat = pickle.load(open("all_dat.pickle", 'rb'))
for key in all_dat:
    print("all_dat", key, all_dat[key].shape)
    

fit_delta_cosmo(all_dat["redshift"], all_dat["delta_mu"], None, all_dat["dmudg"], all_dat["dmudr"], all_dat["dmudi"], all_dat["dmudz"])

