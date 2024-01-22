import glob
import pickle
import gzip
from scipy.stats import scoreatpercentile
import numpy as np
from FileRead import read_param
import sys
import tqdm
import matplotlib.pyplot as plt
import multiprocessing
multiprocessing.set_start_method("fork")
import pystan 



def fmt(val, unc, mean_unc, chi2_DoF):
    if np.isnan(val):
        assert np.isnan(unc)
        return "\\nodata"
    else:
        if type(chi2_DoF) == str:
            chi2_DoF_fmt = chi2_DoF
        elif np.isnan(chi2_DoF):
            chi2_DoF_fmt = "\\nodata"
        else:
            chi2_DoF_fmt = "%.2f" % chi2_DoF


        return "$%.3f \pm %.3f \pm %.3f \ (%s)$" % (val, unc, mean_unc, chi2_DoF_fmt)

def verify_filenamelist(sampfl):
    filenamelist = read_param(sampfl.split("/")[0] + "/paramfile.txt", "filenamelist")
    if suffix == "LH":
        assert filenamelist[0].count("_L_")
        assert filenamelist[1].count("_H_")
        assert filenamelist[2].count("_V_")
    elif suffix == "H":
        assert filenamelist[0].count("_H_")
    else:
        raise "Uknown suffix " + suffix

    

try:
    suffix = sys.argv[1]
except:
    print("Needs suffix like LH or H")
    assert 0
    
if suffix == "H":
    cosmomodel = "1"
else:
    cosmomodel = "5"

datasetkeys = []
    
for tmpind in range(1 + (suffix == "LH")*2):
    datasetkeys.append("mobs_cuts[%i]" % (tmpind + 1))
    datasetkeys.append("mobs_cut_sigmas[%i]" % (tmpind + 1))
    datasetkeys.append("sigma_int[%i]" % (tmpind + 1))

    
pars = ["Om"]*(cosmomodel == "1") + ["wDE", "wpivot15", "waDE"]*(cosmomodel == "5") + ["alpha", "beta_B", "beta_R_low", "beta_R_high", "delta_0", "delta_h"] + datasetkeys  + ["mBx1c_int_variance[1]", "mBx1c_int_variance[2]", "mBx1c_int_variance[3]", "outl_frac"]


labels = {"Om": "$\Omega_m$", "wDE": "$w_0$",
          "wpivot12": "$w_0 + 0.12\;w_a$", "wpivot15": "$w_0 + 0.15\;w_a$", "wpivot18": "$w_0 + 0.18\;w_a$",
          "waDE": "$w_a$", "this_MB": "$\mathcal{M}_B$", "alpha": "$\\alpha$",
          "beta_B": "$\\beta_B$",
          "beta_R_low": "$\\beta_{RL}$",
          "beta_R_high": "$\\beta_{RH}$",
          "delta_0": "$\delta(0)$",
          "delta_h": "$\delta(\infty)/\delta(0)$",
          "mobs_cuts[1]": "$m_{50}$",
          "mobs_cut_sigmas[1]": "$\sigma_m$",
          "sigma_int[1]": "$\sigma^{\mathrm{unexpl}}$",
          "mBx1c_int_variance[1]": "$f^{m_B}$",
          "mBx1c_int_variance[2]": "$f^{x_1}$",
          "mBx1c_int_variance[3]": "$f^{c}$",
          "outl_frac": "$f^{\mathrm{outl}}$"}

true_vals = {"Om": 0.3, "wDE": -1, "waDE": 0, "wpivot12": -1, "wpivot15": -1, "wpivot18": -1,
             "alpha": 0.15, "beta_B": 3.1, "beta_R_low": 3.1, "beta_R_high": 3.1,
             "delta_0": 0.08,
             "delta_h": "$\mathcal{U}(0,\ 1)$",
             "mobs_cuts[1]": "\\nodata",
             "mobs_cut_sigmas[1]": "\\nodata",
             "sigma_int[1]": 0.12,
             "mBx1c_int_variance[1]": "Simplex",
             "mBx1c_int_variance[2]": "Simplex",
             "mBx1c_int_variance[3]": "Simplex",
             "outl_frac": 0}


if suffix == "LH":
    for tmpind, tmpkey in enumerate(["Low", "Mid", "High"]):
        labels["mobs_cuts[%i]" % (tmpind + 1)] = "$m_{50}$ %s-$z$" % tmpkey
        labels["mobs_cut_sigmas[%i]" % (tmpind + 1)] = "$\sigma_m$ %s-$z$" % tmpkey
        labels["sigma_int[%i]" % (tmpind + 1)] = "$\sigma^{\mathrm{unexpl}}$ %s-$z$" % tmpkey

        true_vals["sigma_int[%i]" % (tmpind + 1)] = 0.12
        if tmpkey != "High":
            true_vals["mobs_cuts[%i]" % (tmpind + 1)] = "\\nodata"
            true_vals["mobs_cut_sigmas[%i]" % (tmpind + 1)] = "\\nodata"
        else:
            true_vals["mobs_cuts[%i]" % (tmpind + 1)] = 26.0
            true_vals["mobs_cut_sigmas[%i]" % (tmpind + 1)] = 0.25

"""
for x1c	in ["x1", "c"]:
    for	x1cfmt in ["%s_star", "R_%s", "tau_%s"]:
        for i in range(2):
            LH = "HL"[i]
            the_par = (x1cfmt % x1c) + ("\[%s\]" % (i+1))
            pars.append(the_par)
            if x1cfmt.count("star"):
                labels[the_par] = "$" + x1c.replace("x1", "x") + "_{%s%s}^{*}$" % ("1"*(x1c == "x1"), LH)
            if x1cfmt.count("R"):
                labels[the_par] = "$R^{%s}_%s$" % (x1c.replace("x1", "x_1"), LH)
            if x1cfmt.count("tau"):
                labels[the_par] = "$\\tau_{%s%s}$" % (x1c.replace("x1", "x_1"), LH)
"""

all_txt_grid = []

all_fmB_true = []
all_fmB_posterior = []

for matchstr, description in [
        ("UNITY" + suffix + "_nosel_cos=" + cosmomodel + "_???", "No Selection Effects"),
        ("UNITY" + suffix + "_nosel_twopop_cos=" + cosmomodel + "_???", "No Selection Effects"),
        ("UNITY" + suffix + "_cos=" + cosmomodel + "_???", "Nominal UNITY1.5 Model")]:
        #("UNITY" + suffix + "_1D_???/log.txt", "UNITY1.5, 1D Unexplained"),
        #("UNITY" + suffix + "_nocal_???/log.txt", "UNITY1.5, No $\Delta$sys")

    sampfls = glob.glob(matchstr + "/*samples*pickle")
    

    all_pars = {}
    all_uncs = {}
    all_trues = {}

    for par in pars:
        all_pars[par] = []
        all_uncs[par] = []
        all_trues[par] = []
    
    for sampfl in tqdm.tqdm(sampfls):
        UNITY_paramfl = sampfl.split("/")[0] + "/paramfile.txt"
        sim_paramfl = "params_" + sampfl.split("/")[0].split("_")[-1] + ".dat"
        
        fit_params = pickle.load(gzip.open(sampfl, 'rb'))
        if cosmomodel == "5":
            #fit_params["wpivot12"] = fit_params["wDE"] + 0.12*fit_params["waDE"]
            fit_params["wpivot15"] = fit_params["wDE"] + 0.15*fit_params["waDE"]
            #fit_params["wpivot18"] = fit_params["wDE"] + 0.18*fit_params["waDE"]

        if description == "Nominal UNITY1.5 Model":
            fmB_true = read_param(sim_paramfl, "frac_var_mBx1c")
            assert fmB_true[0] == "["
            fmB_true = fmB_true.replace("[", "").replace("]", "")
            all_fmB_true.append(float(fmB_true))
            all_fmB_posterior.append(fit_params["mBx1c_int_variance"][:,0])
            
        verify_filenamelist(sampfl)
            
        for par in pars:
            if par.count("[") == 1:
                ind = int(par.split("[")[-1].split("]")[0])
                parnoind = par.split("[")[0]
                
                all_pars[par].append(np.median(fit_params[parnoind][:, ind - 1]))
                all_uncs[par].append(0.5*(scoreatpercentile(fit_params[parnoind][:,ind - 1], 84.1345) - scoreatpercentile(fit_params[parnoind][:, ind - 1], 15.8655)))
            else:
                all_pars[par].append(np.median(fit_params[par]))
                all_uncs[par].append(0.5*(scoreatpercentile(fit_params[par], 84.1345) - scoreatpercentile(fit_params[par], 15.8655)))

            try:
                float(true_vals[par])
                all_trues[par].append(true_vals[par])
            except:
                all_trues[par].append(np.sqrt(-1.))
                
            if ["mBx1c_int_variance[1]", "mBx1c_int_variance[2]", "mBx1c_int_variance[3]"].count(par):
                if read_param(UNITY_paramfl, "threeD_unexplained") == 0:
                    if par == "mBx1c_int_variance\[1\]":
                        all_pars[par][-1] = 1
                        all_uncs[par][-1] = 0
                    else:
                        all_pars[par][-1] = 0
                        all_uncs[par][-1] = 0
            
                f_true = str(read_param(sim_paramfl, "frac_var_mBx1c", ind = int(par[-2])))
                f_true = f_true.replace("[", "").replace("]", "")
                all_trues[par][-1] = float(f_true)


            if par.count("mobs_cut"):
                if read_param(sampfl.split("/")[0] + "/paramfile.txt", "stan_code").count("no_sel"):
                    all_pars[par][-1] = np.sqrt(-1)
                    all_uncs[par][-1] = np.sqrt(-1)
                #all_trues[par][-1] = np.sqrt(-1)

            if par == "delta_h":
                all_trues[par][-1] = read_param(sim_paramfl, "delta_h")
                        
    print("all_pars", all_pars, len(all_pars["beta_B"]))
    towrite = [description]
    for par in pars:
        the_mean = np.mean(all_pars[par])
        the_std = np.std(all_pars[par], ddof=1)
        sqrtn = np.sqrt(float(len(all_pars[par])))

        mean_unc = np.mean(all_uncs[par])

        print(par, all_trues[par])
        chi2_DoF = (np.array(all_pars[par]) - np.array(all_trues[par]))/np.array(all_uncs[par])
        chi2_DoF = sum(chi2_DoF**2.)/len(all_pars[par])
            
        towrite.append(fmt(the_mean, the_std/sqrtn, mean_unc = mean_unc, chi2_DoF = chi2_DoF))

        #the_mean = np.mean(all_uncs[par])
        #the_std = np.std(all_uncs[par], ddof=1)

        #towrite.append(fmt(the_mean, the_std/sqrtn))
        
    all_txt_grid.append(towrite)
    

all_txt_grid = np.array(all_txt_grid)    

print("all_txt_grid", all_txt_grid, all_txt_grid.shape)


print("Parameter & Input & " + " & ".join(all_txt_grid[:,0]))
for i in range(len(pars)):
    for valunc in range(1):
        try:
            float(true_vals[pars[i]])
            input_txt = "%.3f" % true_vals[pars[i]]
        except:
            input_txt = true_vals[pars[i]]
            
        towrite = labels[pars[i]] + " & " + input_txt + " & "
        
        for j in range(len(all_txt_grid)):
            towrite += all_txt_grid[j][i+1+valunc]
        
            if j != len(all_txt_grid) -1:
                towrite += " & "
            
        towrite += "\\\\"
        print(towrite)
    


skew_code = """
data {
    int n_samp;
    vector [n_samp] posterior;
}

parameters {
    real <lower = -4, upper = 4> mu;
    real <lower = 0.01, upper = 5> sig;
    real alpha;
}

model {
    posterior ~ skew_normal(mu, sig, alpha);
}
"""

try:
    sk, sc = pickle.load(open("skew.pickle", 'rb'))
    if sc != skew_code:
        time_to_raise
except:
    sk = pystan.StanModel(model_code=skew_code)
    pickle.dump([sm, skew_code], open("skew.pickle", 'wb'))

        
stan_code = """
data {
    int n_obs;

    vector [n_obs] mu;
    vector [n_obs] sig;
    vector [n_obs] alpha;

    vector [n_obs] xvals;
    vector [100] pltx;
}

parameters {
    real slope;
    real intercept;
}

transformed parameters {
    vector [100] plty;
    plty = pltx*slope + intercept;
}

model {
    for (i in 1:n_obs) {
        slope*xvals[i] + intercept ~ skew_normal(mu[i], sig[i], alpha[i]);
    }
}
"""

try:
    sm, sc = pickle.load(open("stan.pickle", 'rb'))
    if sc != stan_code:
        time_to_raise
except:
    sm = pystan.StanModel(model_code=stan_code)
    pickle.dump([sm, stan_code], open("stan.pickle", 'wb'))

stan_data = dict(n_obs = len(all_fmB_posterior), mu = [], sig = [], alpha = [], xvals = all_fmB_true, pltx = np.linspace(0, 1, 100))

plt.figure(figsize = (5, 5))
for i in tqdm.trange(len(all_fmB_posterior)):
    plt.plot([all_fmB_true[i]]*2, scoreatpercentile(all_fmB_posterior[i], [15.8655, 84.1345]), color = 'b', zorder = 1)
    plt.plot(all_fmB_true[i], np.median(all_fmB_posterior[i]), '.', color = 'b', zorder = 1)

    fit = sk.sampling(data=dict(n_samp = len(all_fmB_posterior[i]), posterior = all_fmB_posterior[i]), iter=1000, chains=2, refresh = 1000)
    print(fit)
    fit_params = fit.extract(permuted = True)
    stan_data["mu"].append(np.median(fit_params["mu"]))
    stan_data["sig"].append(np.median(fit_params["sig"]))
    stan_data["alpha"].append(np.median(fit_params["alpha"]))


fit = sm.sampling(data=stan_data, iter=2000, chains=4, refresh = 100)
print(fit)
fit_params = fit.extract(permuted = True)

plt.fill_between(stan_data["pltx"], scoreatpercentile(fit_params["plty"], 15.8655, axis = 0),
                 scoreatpercentile(fit_params["plty"], 84.1345, axis = 0), color = 'b', zorder = -2, alpha = 0.5, label = "$y = %.2f x + %.2f$" % (np.median(fit_params["slope"]),
                                                                                                                                                   np.median(fit_params["intercept"])))
plt.legend(loc = 'best')


plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')

plt.xlabel("Simulation Fraction of Unexplained\nVariance in $m_B$ ($f^{m_B}$)")
plt.ylabel("Posterior Fraction of Unexplained\nVariance in $m_B$ ($f^{m_B}$)")
plt.savefig("f_mB_recovered_" + suffix + ".pdf", bbox_inches = 'tight')
plt.close()
