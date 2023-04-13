from FileRead import readcol, read_param
import glob
import matplotlib.pyplot as plt
import numpy as np
from subprocess import getoutput

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


for resfl in glob.glob("dataset_000/*/res*salt2.dat"):
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


plt.figure(figsize = (24, 16))
for i, keys in enumerate([("redshift", "delta_c"),
                          ("redshift", "delta_x1"),
                          ("redshift", "pulls_c"),
                          ("redshift", "pulls_x1"),
                          ("redshift", "delta_mu"),
                          ("redshift", "delta_mag"),
                          ("dmudg", "delta_mu"),
                          ("dmudr", "delta_mu"),
                          ("dmudi", "delta_mu"),
                          ("dmudz", "delta_mu"),
                          ("true_c", "delta_c"),
                          ("true_x1", "delta_x1")]):
    
    plt.subplot(4,3,i+1)
    plt.scatter(all_dat[keys[0]], all_dat[keys[1]], label = "Mean %.3f +- %.3f Median %.3f RMS %.3f" % (np.mean(all_dat[keys[1]]), np.std(all_dat[keys[1]], ddof=1)/np.sqrt(float(len(all_dat[keys[1]]))),
                                                                                                        np.median(all_dat[keys[1]]),
                                                                                                        np.std(all_dat[keys[1]], ddof=1)))
    plt.axhline(0)
    plt.legend(loc = 'best')
    
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])

plt.savefig("compare_LC_vs_input.pdf")
plt.close()

