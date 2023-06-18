from FileRead import readcol, read_param
import glob
import matplotlib.pyplot as plt
import numpy as np
from subprocess import getoutput
import tqdm
import sys
from scipy.stats import scoreatpercentile

def do_med_bins(x, y, nbins, average_not_median = 0):
    binedges = scoreatpercentile(x, np.linspace(0, 100, nbins+1))
    binedges[0] -= (binedges[1] - binedges[0])*0.001
    binedges[-1] += (binedges[-1] - binedges[-2])*0.001

    binx = []
    biny = []
    
    for i in range(nbins):
        inds = np.where((x >= binedges[i])*(x < binedges[i+1]))

        if average_not_median:
            binx.append(np.mean(x[inds]))
            biny.append(np.mean(y[inds]))
        else:
            binx.append(np.median(x[inds]))
            biny.append(np.median(y[inds]))
    return binx, biny

def get_dmu(band, resfl):
    cmd = "grep Zeropoint " + resfl.replace("result_salt2.dat", "result_deriv.dat") + " | grep 'AB|SDSS|SDSS_" + band + "'"
    print(cmd)
    
    grepout = getoutput(cmd)

    try:
        return float(grepout.split(None)[4])
    except:
        return 0.

    
all_dat = dict(true_c = [], delta_c = [], obs_sig_c = [],
               true_x1 = [], delta_x1 = [], obs_sig_x1 = [],
               delta_mag = [],
               delta_mu = [],
               dmudg = [],
               dmudr = [],
               dmudi = [],
               dmudz = [],
               redshift = [])


if len(sys.argv) > 1:
    globstr = "dataset_*/*00*/res*salt2.dat"
else:
    globstr = "dataset_*/*/res*salt2.dat"

for resfl in tqdm.tqdm(glob.glob(globstr)):
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

        
        all_dat["true_c"].append(true_c)
        all_dat["delta_c"].append(obs_c - true_c)
        all_dat["obs_sig_c"].append(obs_sig_c)
        

        all_dat["delta_mag"].append(obs_mag - true_mag)

        all_dat["true_x1"].append(true_x1)
        all_dat["delta_x1"].append(obs_x1 - true_x1)
        all_dat["obs_sig_x1"].append(obs_sig_x1)


        all_dat["delta_mu"].append((obs_mag + 0.14*obs_x1 - 3.1*obs_c) - (true_mag + 0.14*true_x1 - 3.1*true_c))
        
        all_dat["redshift"].append(redshift)

        for band in "griz":
            all_dat["dmud" + band].append(get_dmu(band, resfl))

for key in all_dat:
    all_dat[key] = np.array(all_dat[key])

all_dat["pulls_c"] = all_dat["delta_c"]/all_dat["obs_sig_c"]
all_dat["pulls_x1"] = all_dat["delta_x1"]/all_dat["obs_sig_x1"]


plt.figure(figsize = (24, 24))
for i, keys in enumerate([("redshift", "delta_mag", 0),
                          ("redshift", "delta_c", 0),
                          ("true_c", "delta_c", 0),
                          ("redshift", "delta_x1", 0),
                          ("redshift", "pulls_c", 0),
                          ("redshift", "pulls_x1", 0),
                          ("redshift", "delta_mu", 0),
                          ("redshift", "delta_mag", 1),
                          ("redshift", "delta_c", 1),
                          ("true_c", "delta_c", 1),
                          ("redshift", "delta_x1", 1),
                          ("redshift", "delta_mu", 1),
                          ("dmudg", "delta_mu", 0),
                          ("dmudr", "delta_mu", 0),
                          ("dmudi", "delta_mu", 0),
                          ("dmudz", "delta_mu", 0),
                          ("true_x1", "delta_x1", 0)]):
    
    plt.subplot(4,5,i+1)
    if keys[2] == 0:
        plt.scatter(all_dat[keys[0]], all_dat[keys[1]], label = "Mean %.3f +- %.3f Median %.3f RMS %.3f" % (np.mean(all_dat[keys[1]]), np.std(all_dat[keys[1]], ddof=1)/np.sqrt(float(len(all_dat[keys[1]]))),
                                                                                                            np.median(all_dat[keys[1]]),
                                                                                                            np.std(all_dat[keys[1]], ddof=1)))
    else:
        binx, biny = do_med_bins(all_dat[keys[0]], all_dat[keys[1]], 15)
        plt.plot(binx, biny, '.', color = 'k')

        binx, biny = do_med_bins(all_dat[keys[0]], all_dat[keys[1]], 15, average_not_median = 1)
        plt.plot(binx, biny, '^', color = 'g')

    plt.axhline(0)
    plt.legend(loc = 'best')
    
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])

    if keys[0] == "redshift":
        plt.xscale('log')

plt.tight_layout()
plt.savefig("compare_LC_vs_input.pdf", bbox_inches = 'tight')
plt.close()

