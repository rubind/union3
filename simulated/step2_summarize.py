import glob
try:
    from commands import getoutput
except:
    from subprocess import getoutput

import numpy as np

def fmt(val, unc):
    return "%.3f +- %.3f" % (val, unc)


pars = ["Om", "alpha", "beta_B", "beta_R_low", "beta_R_high", "delta_0", "delta_h", "mobs_cuts\[1\]", "mobs_cut_sigmas\[1\]"]

for matchstr, description in [
        ("UNITY_stan_code_simple_copy2???/log.txt", "Default"),
        ("UNITY_stan_code_simple_nosel*/log.txt", "No Selection Effects"),
        ("stan_code_simple_tight_calib*/log.txt", "No Calib"),
        ("UNITY_stan_code_simple_copy???/log.txt", "One beta"),
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

    print(" & ".join(towrite))
