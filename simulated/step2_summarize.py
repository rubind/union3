import glob
import pickle
import gzip
from scipy.stats import scoreatpercentile
import numpy as np
from FileRead import read_param, readcol
import sys
import tqdm
import matplotlib.pyplot as plt
import multiprocessing
multiprocessing.set_start_method("fork")
import pystan 
import matplotlib.pyplot as plt



def fmt(val, unc, mean_unc, chi2_DoF):
    if np.isnan(val):
        assert np.isnan(unc)
        return "\\nodata"
    else:
        if type(chi2_DoF) == str:
            chi2_DoF_fmt = chi2_DoF
        elif np.isnan(chi2_DoF):
            chi2_DoF_fmt = "$\\nodata$"
        else:
            chi2_DoF_fmt = "%.2f" % chi2_DoF


        return "$%.3f \pm %.3f \pm %.3f \ (%s)$" % (val, unc, mean_unc, chi2_DoF_fmt)

def verify_filenamelist(sampfl):
    filenamelist = read_param(sampfl.split("/")[0] + "/paramfile.txt", "filenamelist")
    if suffix == "LH":
        assert filenamelist[0].count("_S_")
        assert filenamelist[1].count("_L_")
        assert filenamelist[2].count("_H_")
        assert filenamelist[3].count("_V_")
    elif suffix == "H":
        assert filenamelist[0].count("_H_")
    else:
        raise "Uknown suffix " + suffix

    #[v1s, total, outliers] = readcol("outliers_by_dataset.txt", 'aff')
    
    """
    total_SNe = 0.
    total_outliers = 0.
    
    for flnm in filenamelist:
        ind = v1s.index(flnm.split("/")[-1])
        
        total_SNe += total[ind]
        total_outliers += outliers[ind]
    """

    """
    #SN	RA	DEC	ZHEL	ZCMB	PASS
    dataset_S_010/SNSI0001	0.0	0.0	0.014905	0.014905	True
    dataset_S_010/SNSI0009	0.0	0.0	0.011504	0.011504	True
    """

    [SNname, NA, NA, NA, NA, SNpass] = readcol(sampfl.split("/")[0] + "/sn_input.txt", 'affffa')

    total_outliers = 0.
    total_SNe = 0.

    for i in range(len(SNname)):
        if SNpass[i] == "True":
            total_SNe += 1
            SNparsed = SNname[i].split("/")[-1]
            assert SNparsed[:2] == "SN"
            assert SNname[i].split("_")[1] == SNparsed[2]
            if SNparsed[3] == "I":
                pass
            elif SNparsed[3] == "O":
                total_outliers += 1
            else:
                assert 0, SNparsed
    
    return total_outliers/total_SNe

def plot_cosmetics(make_symm, true_val):
    plt.ylim(0, 1)
    plt.yticks([])
    plt.tick_params(axis='x', direction='in')
    for label in plt.gca().get_xticklabels():
        label.set_verticalalignment('bottom')
        label.set_horizontalalignment('center')
        label.set_y(0.1)

    if make_symm:
        xlim = plt.xlim()
        dx = max(xlim[1] - true_val, true_val - xlim[0])
        plt.xlim(true_val - dx, true_val + dx)


def check_if_blank(ax):
    is_blank = True

    # Checking for line plots
    for line in ax.lines:
        if line.get_xdata().size > 0:
            return False
            

    # Checking for patch objects (e.g., bars in a bar chart)
    for patch in ax.patches:
        return False
    return True

        
def bbox_subplot(i, j, pars, fig, add_label = 0):
    pad_y = 0.02
    pad_x = 0.05

    y = (1. - pad_y*3)/len(pars)
    x = (1. - pad_x*5)/4.

    ax = plt.axes([pad_x + j*(pad_x + x),
                   pad_y + ((len(pars) - 1) - i)*y + pad_y*(pars[:i+1].count("alpha") == 0),
                   x, y])

    if add_label:
        fig.text(0.5, 1.0 + pad_y/5., "Cosmology Parameters", ha = 'center', va = 'center', fontsize = 12)
        fig.text(0.5, 1 - (pars.index("alpha"))*y - pad_y*1.5, "Other Parameters", ha = 'center', va = 'center', fontsize = 12)
    
    return ax

matchstrs = sys.argv[1:]

for matchstr in matchstrs:
    if matchstrs[0].count("UNITYLH"):
        suffix = "LH"
    else:
        suffix = "H"
    assert matchstrs[0].count("UNITYLH") == matchstr.count("UNITYLH")
    
    cosmomodel = matchstrs[0].split("cos=")[1][0]
    assert matchstrs[0].split("cos=")[1][0] == matchstr.split("cos=")[1][0]

    

datasetkeys = []
    
for tmpind in range(1 + (suffix == "LH")*3):
    datasetkeys.append("mobs_cuts[%i]" % (tmpind + 1))
    datasetkeys.append("mobs_cut_sigmas[%i]" % (tmpind + 1))
    datasetkeys.append("sigma_int[%i]" % (tmpind + 1))

poppars = []
for x1c in ["x1", "c"]:
    for fast_slow in ["fast", "slow"]:
        poppars.append(x1c + "_star_" + fast_slow)
        poppars.append("R_" + x1c + "_" + fast_slow)

    
pars = ["H0"] + ["Om"]*(cosmomodel == "1") + ["wDE", "wpivot15", "waDE"]*(cosmomodel == "5") + ["alpha", "alpha_fast", "alpha_slow", "beta_B", "beta_R_low", "beta_R_high", "delta_0", "delta_h", "step_mass", "MB_fast_minus_slow"] + poppars  + datasetkeys  + ["sigma_int_fast", "mBx1c_int_variance[1]", "mBx1c_int_variance[2]", "mBx1c_int_variance[3]", "outl_frac"]


dir_labels = {"UNITY" + suffix + "twox1_cos=" + cosmomodel + "_": "UNITY1.8, Two-$x_1$ Modes",
              "UNITY" + suffix + "twox1_cos=" + cosmomodel + "_nooutl_noutlmod_": "1.8, No Outl, No Outl Model",
              "UNITY" + suffix + "_cos=" + cosmomodel + "_": "UNITY1.7"}


labels = {"H0": "$H_0$",
          "Om": "$\Omega_m$", "wDE": "$w_0$",
          "wpivot12": "$w_0 + 0.12\;w_a$", "wpivot15": "$w_0 + 0.15\;w_a$", "wpivot18": "$w_0 + 0.18\;w_a$",
          "waDE": "$w_a$", "this_MB": "$\mathcal{M}_B$", "MB_fast_minus_slow": "$\mathcal{M}_B$ fast $-$ slow",
          "alpha": "$\\alpha$", "alpha_fast": "$\\alpha$ fast", "alpha_slow": "$\\alpha$ slow",
          "beta_B": "$\\beta_B$",
          "beta_R_low": "$\\beta_{RL}$",
          "beta_R_high": "$\\beta_{RH}$",
          "step_mass": "step mass",
          "delta_0": "$\delta(0)$",
          "delta_h": "$\delta(\infty)/\delta(0)$",
          "x1_star_fast": "$x_1^*$ Fast",
          "x1_star_slow": "$x_1^*$ Slow",
          "R_x1_fast": "$R^{x_1}$ Fast",
          "R_x1_slow": "$R^{x_1}$ Slow",
          "c_star_fast": "$c^*$ Fast",
          "c_star_slow": "$c^*$ Slow",
          "R_c_fast": "$R^c$ Fast",
          "R_c_slow": "$R^c$ Slow",

          "mobs_cuts[1]": "$m_{50}$",
          "mobs_cut_sigmas[1]": "$\sigma_m$",
          "sigma_int[1]": "$\sigma^{\mathrm{unexpl}}$",
          "sigma_int_fast": "$\sigma^{\mathrm{unexpl}}$ fast",
          "mBx1c_int_variance[1]": "$f^{m_B}$",
          "mBx1c_int_variance[2]": "$f^{x_1}$",
          "mBx1c_int_variance[3]": "$f^{c}$",
          "outl_frac": "$f^{\mathrm{outl}}$"}

true_vals = {"H0": 71, "Om": 0.3, "wDE": -1, "waDE": 0, "wpivot12": -1, "wpivot15": -1, "wpivot18": -1,
             "MB_fast_minus_slow": -0.14,
             "beta_B": 2.1, "beta_R_low": 4.4, "beta_R_high": 3.2,
             "step_mass": 10.,
             "delta_0": 0.0,
             "delta_h": "$\mathcal{U}(0,\ 1)$",
             "mobs_cuts[1]": "\\nodata",
             "mobs_cut_sigmas[1]": "\\nodata",
             "sigma_int[1]": 0.08,
             "sigma_int_fast": 0.08,
             "mBx1c_int_variance[1]": "Simplex",
             "mBx1c_int_variance[2]": "Simplex",
             "mBx1c_int_variance[3]": "Simplex",
             "outl_frac": 0}


if suffix == "LH":
    for tmpind, tmpkey in enumerate(["Very Low", "Low", "Mid", "High"]):
        labels["mobs_cuts[%i]" % (tmpind + 1)] = "$m_{50}$ %s-$z$" % tmpkey
        labels["mobs_cut_sigmas[%i]" % (tmpind + 1)] = "$\sigma_m$ %s-$z$" % tmpkey
        labels["sigma_int[%i]" % (tmpind + 1)] = "$\sigma^{\mathrm{unexpl}}$ %s-$z$" % tmpkey

        true_vals["sigma_int[%i]" % (tmpind + 1)] = 0.08
        #if tmpkey != "High":
        true_vals["mobs_cuts[%i]" % (tmpind + 1)] = "\\nodata"
        true_vals["mobs_cut_sigmas[%i]" % (tmpind + 1)] = "\\nodata"
        #else:
        #    true_vals["mobs_cuts[%i]" % (tmpind + 1)] = 26.0
        #    true_vals["mobs_cut_sigmas[%i]" % (tmpind + 1)] = 0.25
        # The problem with these nominal numbers is that they ignore the effect of cadence.
        
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

"""
fig = plt.figure(figsize = (12, 0.8*len(pars)))

all_ax = []
for i in range(len(pars)):
    all_ax.append([])
    for j in range(4):
        all_ax[i].append(
            bbox_subplot(i = i, j = j, pars = pars, fig = fig, add_label = (i == 0)*(j == 0))
        )
"""



tmp_ind = 0

for matchstr in matchstrs:
        #("UNITY" + suffix + "_fixed_cos=" + cosmomodel + "_???", "Improved Outlier Limits"),
        #("UNITY" + suffix + "_nosel_cos=" + cosmomodel + "_???", "No Selection Effects"),
        #("UNITY" + suffix + "_nosel_twopop_cos=" + cosmomodel + "_???", "No Sel. Eff., $z$-Dep. Pop.")]:
        #("UNITY" + suffix + "_1D_???/log.txt", "UNITY1.5, 1D Unexplained"),
        #("UNITY" + suffix + "_nocal_???/log.txt", "UNITY1.5, No $\Delta$sys")

    sampfls = glob.glob(matchstr + "???/*samples*pickle")
    description = dir_labels[matchstr]

    all_pars = {}
    all_uncs = {}
    all_uncs_up = {}
    all_uncs_down = {}
    all_trues = {}
    #all_posterior_stacks = {}

    for par in pars:
        all_pars[par] = []
        all_uncs[par] = []
        all_uncs_up[par] = []
        all_uncs_down[par] = []
        all_trues[par] = []
        #all_posterior_stacks[par] = []

    
    
    for sampfl in tqdm.tqdm(sampfls):
        UNITY_paramfl = sampfl.split("/")[0] + "/paramfile.txt"
        sim_paramfl = "params_" + sampfl.split("/")[0].split("_")[-1] + ".dat"
        
        fit_params = pickle.load(gzip.open(sampfl, 'rb'))
        if cosmomodel == "5":
            #fit_params["wpivot12"] = fit_params["wDE"] + 0.12*fit_params["waDE"]
            fit_params["wpivot15"] = fit_params["wDE"] + 0.15*fit_params["waDE"]
            #fit_params["wpivot18"] = fit_params["wDE"] + 0.18*fit_params["waDE"]

        if tmp_ind == 0:
            fmB_true = read_param(sim_paramfl, "frac_var_mBx1c")
            assert fmB_true[0] == "["
            fmB_true = fmB_true.replace("[", "").replace("]", "")
            all_fmB_true.append(float(fmB_true))
            all_fmB_posterior.append(fit_params["mBx1c_int_variance"][:,0])

            for true_key in ["alpha", "alpha_fast", "alpha_slow", "x1_star_fast", "x1_star_slow", "R_x1_fast", "R_x1_slow", "c_star_fast", "c_star_slow", "R_c_fast", "R_c_slow"]:
                try:
                    true_vals[true_key] = float(read_param(sim_paramfl, true_key.replace("R_", "R")))
                except:
                    true_vals[true_key] = np.sqrt(-1.)
            
        true_outl_frac = verify_filenamelist(sampfl)
            
        for par in pars:
            if par.count("[") == 1:
                ind = int(par.split("[")[-1].split("]")[0])
                parnoind = par.split("[")[0]

                if parnoind in fit_params:
                    all_pars[par].append(np.median(fit_params[parnoind][:, ind - 1]))
                    all_uncs[par].append(0.5*(scoreatpercentile(fit_params[parnoind][:,ind - 1], 84.1345) - scoreatpercentile(fit_params[parnoind][:, ind - 1], 15.8655)))
                    all_uncs_up[par].append(scoreatpercentile(fit_params[parnoind][:,ind - 1], 84.1345) - np.median(fit_params[parnoind][:, ind - 1]))
                    all_uncs_down[par].append(np.median(fit_params[parnoind][:, ind - 1]) - scoreatpercentile(fit_params[parnoind][:,ind - 1], 15.8655))
                    #all_posterior_stacks[par].extend(fit_params[parnoind][:, ind - 1])
                else:
                    all_pars[par].append(np.sqrt(-1.))
                    all_uncs[par].append(np.sqrt(-1.))
                    all_uncs_up[par].append(np.sqrt(-1.))
                    all_uncs_down[par].append(np.sqrt(-1.))
                    #all_posterior_stacks[par].extend(np.sqrt(-1.))
            else:
                if par in fit_params:
                    all_pars[par].append(np.median(fit_params[par]))
                    all_uncs[par].append(0.5*(scoreatpercentile(fit_params[par], 84.1345) - scoreatpercentile(fit_params[par], 15.8655)))
                    all_uncs_up[par].append(scoreatpercentile(fit_params[par], 84.1345) - np.median(fit_params[par]))
                    all_uncs_down[par].append(np.median(fit_params[par]) - scoreatpercentile(fit_params[par], 15.8655))
                    #all_posterior_stacks[par].extend(fit_params[par])
                else:
                    all_pars[par].append(np.sqrt(-1.))
                    all_uncs[par].append(np.sqrt(-1.))
                    all_uncs_up[par].append(np.sqrt(-1.))
                    all_uncs_down[par].append(np.sqrt(-1.))
                    #all_posterior_stacks[par].extend(np.sqrt(-1.))

                    

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

                    all_uncs_up[par][-1] = np.sqrt(-1)
                    all_uncs_down[par][-1] = np.sqrt(-1)

                #all_trues[par][-1] = np.sqrt(-1)

            if par == "delta_h":
                all_trues[par][-1] = read_param(sim_paramfl, "delta_h")
            if par == "outl_frac":
                all_trues[par][-1] = true_outl_frac
                
    print("all_pars", all_pars, len(all_pars["beta_B"]))
    towrite = [description]
    for i, par in enumerate(pars):
        #plt.subplot(len(pars), 4, 4*i + 1)
        #plt.sca(all_ax[i][0])
        #if i == 0:
        #    plt.title("Stacked Posterior")
        
        #lower_to_plot = scoreatpercentile(all_posterior_stacks[par], 15.8655)
        #upper_to_plot = scoreatpercentile(all_posterior_stacks[par], 84.1345)
        #med_to_plot = np.median(all_posterior_stacks[par])

        #the_color = ['k', 'b', 'r', 'g'][tmp_ind]
        #the_linewidth = [2,2,1,1][tmp_ind]
        


        #yval = 0.9 - tmp_ind*0.2 #0.125 + tmp_ind*0.25
        #plt.plot(med_to_plot, yval, '.', color = the_color)
        #plt.plot([lower_to_plot, upper_to_plot], [yval]*2, label = description, color = the_color)


        #if np.isclose(np.std(all_trues[par]), 0):
        #    if tmp_ind == 0:
        #        plt.axvline(all_trues[par][0], color = 'k')

        #if labels[par].count(" ") < 2:
        #    plt.ylabel(labels[par].replace(" ", '\n'), rotation = 0, horizontalalignment='right', verticalalignment = 'center', fontsize = 12)
        #else:
        #    plt.ylabel(labels[par], rotation = 0, horizontalalignment='right', verticalalignment = 'center', fontsize = 12)
            
        #plot_cosmetics(make_symm = np.isclose(np.std(all_trues[par]), 0)*(tmp_ind == 3),
        #               true_val = all_trues[par][0])
        
        #plt.hist(all_posterior_stacks[par], alpha = 0.5, label = description)
        
        the_mean = np.mean(all_pars[par])
        the_std = np.std(all_pars[par], ddof=1)
        sqrtn = np.sqrt(float(len(all_pars[par])))

        mean_unc = np.mean(all_uncs[par])

        print(par, all_trues[par])
        pulls = (np.array(all_pars[par]) - np.array(all_trues[par]))/np.array(all_uncs[par])
        chi2_DoF = np.sqrt(sum(pulls**2. / len(all_pars[par])))
        #chi2_DoF = sum(chi2_DoF**2.)/len(all_pars[par])
        
        towrite.append(fmt(the_mean, the_std/sqrtn, mean_unc = mean_unc, chi2_DoF = chi2_DoF))

        #plt.sca(all_ax[i][1])
        #plt.subplot(len(pars), 4, 4*i + 2)

        #if i == 0:
        #    plt.title("Stacked Pulls")

        pulls = []
        for j in range(len(all_pars[par])):
            if all_trues[par][j] > all_pars[par][j]:
                pulls.append(   (all_pars[par][j] - all_trues[par][j])/all_uncs_up[par][j]   )
            else:
                pulls.append(   (all_pars[par][j] - all_trues[par][j])/all_uncs_down[par][j]   )

        
        lower_to_plot = scoreatpercentile(pulls, 15.8655)
        upper_to_plot = scoreatpercentile(pulls, 84.1345)
        med_to_plot = np.median(pulls)

        """
        plt.plot(med_to_plot, yval, '.', color = the_color)
        plt.plot([lower_to_plot, upper_to_plot], [yval]*2, label = description, color = the_color)

        span_color = (0.7, 0.7, 0.7)
        
        if tmp_ind == 0 and np.all(1 - np.isnan(pulls)):
            plt.axvspan(-np.sqrt(np.pi/(2.*len(pulls))),
                        np.sqrt(np.pi/(2.*len(pulls))), color=span_color, zorder = -1)

            plt.axvspan(-1 - 0.1316218*np.sqrt(100./len(pulls)),
                        -1 + 0.1646250*np.sqrt(100./len(pulls)),
                        color=span_color, zorder = -1)

            plt.axvspan(1 - 0.1646250*np.sqrt(100./len(pulls)),
                        1 + 0.1316218*np.sqrt(100./len(pulls)),
                        color=span_color, zorder = -1)
                         
        plot_cosmetics(make_symm = (tmp_ind == 3), true_val = 0)
        """

        #plt.subplot(len(pars), 4, 4*i + 3)
        
        #plt.sca(all_ax[i][2])
        #if i ==	0:
        #    plt.title("Mean(Pulls)")

        
        #plt.plot(np.mean(pulls), yval, '.', color = the_color)
        #plot_cosmetics(make_symm = (tmp_ind == 3), true_val = 0)

        #if tmp_ind == 0 and np.all(1 - np.isnan(pulls)):
        #    plt.axvspan(- 1./np.sqrt(1.*len(pulls)),
        #                1./np.sqrt(1.*len(pulls)), color=span_color, zorder = -1)

        
        #plt.subplot(len(pars), 4, 4*i + 4)

        #plt.sca(all_ax[i][3])
        #if i == 0:
        #    plt.title("RMS(Pulls)")

        #plt.plot(np.std(pulls, ddof=1), yval, '.', color = the_color, label = description*(i == 0))

        #if tmp_ind == 0 and np.all(1 - np.isnan(pulls)):
        #    plt.axvspan(1 - 1./np.sqrt(2.*len(pulls)),
        #                1 + 1./np.sqrt(2.*len(pulls)), color=span_color, zorder = -1)
            
        
        #plot_cosmetics(make_symm = (tmp_ind == 3), true_val = 1)


        #the_mean = np.mean(all_uncs[par])
        #the_std = np.std(all_uncs[par], ddof=1)

        #towrite.append(fmt(the_mean, the_std/sqrtn))
        
    all_txt_grid.append(towrite)
    tmp_ind += 1

"""
for j in range(1,4):
    xlim = [100, -100]
    for i in range(len(pars)):
""" 
    
#for i, par in enumerate(pars):
#plt.sca(all_ax[0][3])
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

            
#plt.subplots_adjust(hspace = 0)
#plt.tight_layout()
#plt.savefig("sim_parameters_" + suffix + ".pdf", bbox_inches = 'tight')
#plt.close()


all_txt_grid = np.array(all_txt_grid)    

print("all_txt_grid", all_txt_grid, all_txt_grid.shape)


print("Parameter & Input & " + " & ".join(all_txt_grid[:,0]))

print("\hline % &")
print("\multicolumn{" + str(len(matchstrs) + 2) + "}{c}{Cosmology Parameters}\\\\ % &")
print("\hline % &")


for i in range(len(pars)):
    if pars[i] == "alpha":
        print("\hline % &")
        print("\multicolumn{" + str(len(matchstrs) + 2) + "}{c}{Other Parameters}\\\\ % &")
        print("\hline % &")
        
    for valunc in range(1):
        try:
            float(true_vals[pars[i]])
            input_txt = "$%.3f$" % true_vals[pars[i]]
        except:
            input_txt = true_vals[pars[i]]

        if pars[i] == "outl_frac":
            input_txt = "%.3f--%.3f" % (min(all_trues[par]), max(all_trues[par]))
            
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

plt.figtext(0.91, 0.93, "Simulated Data", color = 'r', ha = 'right', va = 'top', bbox=dict(edgecolor = 'r', pad = 1, facecolor = 'w'))

plt.savefig("f_mB_recovered_" + suffix + ".pdf", bbox_inches = 'tight')
plt.close()
