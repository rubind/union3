import glob
try:
    from commands import getoutput
except:
    from subprocess import getoutput

import numpy as np

def fmt(val, unc):
    return "%.3f +- %.3f" % (val, unc)


pars = ["Om", "alpha", "beta_B", "beta_R_low", "beta_R_high", "delta_0", "delta_h", "mobs_cuts\[1\]", "mobs_cut_sigmas\[1\]", "sigma_int\[1\]", "sig_int_vector\[1,1\]", "sig_int_vector\[1,2\]", "sig_int_vector\[1,3\]"]

all_txt_grid = []

for matchstr, description in [
        ("UNITY_nosel_???/log.txt", "No Selection Effects"),
        ("UNITY_???/log.txt", "Nominal UNITY1.5 Model"),
        ("UNITY_1D_???/log.txt", "UNITY1.5, 1D Unexplained"),
        ("UNITY_nocal_???/log.txt", "UNITY1.5, No $\Delta$sys")
]:
    logs = glob.glob(matchstr)
    

    all_pars = {}

    for par in pars:
        all_pars[par] = []

        for log in logs:
            val = getoutput("grep '" + par + " ' " + log)
            try:
                val = float(val.split(None)[1])
                all_pars[par].append(val)
            except:
                print("SKIPPING", val, log, par)

    
    print("all_pars", all_pars, len(all_pars["Om"]))
    towrite = [description]
    for par in pars:
        the_mean = np.mean(all_pars[par])
        the_std = np.std(all_pars[par], ddof=1)
        sqrtn = np.sqrt(float(len(all_pars[par])))
        
        towrite.append(fmt(the_mean, the_std/sqrtn))

    all_txt_grid.append(towrite)
    

for i in range(len(all_txt_grid[0])):
    towrite = ""
    
    for j in range(len(all_txt_grid)):
        towrite += all_txt_grid[j][i]

        if j != len(all_txt_grid) -1:
            towrite += " & "
    towrite += "\\\\"
    print(towrite)
    
