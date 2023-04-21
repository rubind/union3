import glob
try:
    from commands import getoutput
except:
    from subprocess import getoutput

import numpy as np

def fmt(val, unc):
    return "$%.3f \pm %.3f$" % (val, unc)


pars = ["Om", "alpha", "beta_B", "beta_R_low", "beta_R_high", "delta_0", "delta_h", "mobs_cuts\[1\]", "mobs_cut_sigmas\[1\]", "sigma_int\[1\]", "mBx1c_int_variance\[1\]", "mBx1c_int_variance\[2\]", "mBx1c_int_variance\[3\]"]

labels = {"Om": "$\Omega_m$", "alpha": "$\alpha$",
          "beta_B": "$\beta_B$",
          "beta_R_low": "$\beta_{RL}$",
          "beta_R_high": "$\beta_{RH}$",
          "delta_0": "$\delta(0)$",
          "delta_h": "$\delta(\infty)/\delta(0)$",
          "mobs_cuts\[1\]": "$m_{50}$",
          "mobs_cut_sigmas\[1\]": "$\sigma_m$",
          "sigma_int\[1\]": "\sigma^{\mathrm{unexpl}}$",
          "mBx1c_int_variance\[1\]": "$f^{\m_B}$",
          "mBx1c_int_variance\[2\]": "$f^{\x_1}$",
          "mBx1c_int_variance\[3\]": "$f^{\c}$"}

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
    towrite = labels[pars[i]] + " & "
    
    for j in range(len(all_txt_grid)):
        towrite += all_txt_grid[j][i]

        if j != len(all_txt_grid) -1:
            towrite += " & "
    towrite += "\\\\"
    print(towrite)
    
