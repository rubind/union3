from numpy import * # I know...
import numpy as np
import pickle as pickle
from matplotlib import use
use("PDF")
import corner # I know...
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile, percentileofscore
import sys
import subprocess
import gzip
from FileRead import readcol
import time
import os
from DavidsNM import miniNM_new

plt.rcParams["font.family"] = "serif"

def get_sample_names_colors():
    f = open(os.environ["UNITY"] + "/paramfiles/sample_names_colors.txt", 'r')
    lines = f.read()
    f.close()

    for i in range(20):
        lines = lines.replace("\t\t", "\t").replace(" \t", "\t").replace("\t ", "\t").replace("  ", " ")
    
    lines = lines.split('\n')
    names_colors_dict = {}
    for line in lines:
        parsed = line.split('\t')
        if len(parsed) > 1:
            names_colors_dict[parsed[0]] = (parsed[1], parsed[2])
    return names_colors_dict


def get_label(key, sh, j=None, k=None):
    print("get_label", key, sh, j, k)
    
    if len(sh) == 2:
        try:
            return key + "_" + str(label_dict.get(key, [j]*(j + 1))[j]
                                   )
        except:
            return key
        
    elif len(sh) == 3:
        return key + "_" + str(label_dict.get(key, [[(j,k)]*(k + 1)]*(j + 1))[j][k]
        )
    else:
        return key


def make_corner(keys, pltname):
    print("Corner plot for ", keys)
    samples = []
    labels = []

    for key in keys:
        if key in fit_params:
            sh = fit_params[key].shape

            if len(sh) == 2:
                for j in range(sh[1]):
                    samples.append(fit_params[key][:,j])
                    labels.append(get_label(key, sh, j=j))
            elif len(sh) == 3:
                for j in range(sh[1]):
                    for k in range(sh[2]):
                        samples.append(fit_params[key][:,j,k])
                        labels.append(get_label(key, sh, j=j, k=k))
            else:
                samples.append(fit_params[key])
                labels.append(get_label(key, sh))
        else:
            print("Skipping ", key)
            
    samples = transpose(array(samples))

    print(samples.shape, labels)

    corner.corner(samples, labels = labels)
    plt.savefig(resdir + pltname)
    plt.close()

def make_kde_corner(keys, pltname):
    labels = []
    samples = []
    latex_labels = {"delta_0": "Host-Mass\nLuminosity Offset $\delta_0$", "delta_beta_R": "Host-Mass\nColor Relation Offset $\delta_{\\beta}$"}
    
    for key in keys:
        samples.append(fit_params[key])
        labels.append(latex_labels[key])
    kde_corner.kde_corner(np.array([samples]), labels, bw_method = 0.2)
    plt.savefig(resdir + pltname, bbox_inches = 'tight')
    plt.close()

    
def make_calib_corner(calib_keys, pltname):
    samples = []
    
    for calib_key in calib_keys:
        ind = the_data["calib_names"].index(calib_key)
        samples.append(fit_params["calibs"][:,ind])


    for i in range(len(fit_params["mobs_cuts"][0])):
        samples.append(fit_params["mobs_cuts"][:,i])
        calib_keys.append(label_dict["mobs_cuts"][i])
        
    samples = transpose(array(samples))


    corner.corner(samples, labels = calib_keys)
    plt.savefig(resdir + pltname)
    plt.close()

    
def make_mB_vs_z():
    plt.subplot(2,1,1)
    plt.plot(stan_data["redshifts"], array(stan_data["obs_mBx1c"])[:,0], '.', color = 'k')
    plt.subplot(2,1,2)
    plt.hist(array(stan_data["obs_mBx1c"])[:,0])
    plt.savefig(resdir + "mB_vs_z.pdf")
    plt.close()


def show_x1color_pop(log_scale):
    plt.figure(figsize = (16,48))

    fit_params["mean_c_by_SN"] = fit_params["c_star_by_SN"] + fit_params["tau_c_by_SN"]

    for i, x1key, ckey, x1label, clabel in zip(list(range(1,6)),
                                               ["x1_star_by_SN", "x1_star_by_SN", "x1_star_by_SN", "R_x1_by_SN", None],
                                               ["mean_c_by_SN", "mean_c_by_SN", "c_star_by_SN", "R_c_by_SN", "tau_c_by_SN"],
                                               ["$x^{*}_1$", "$x^{*}_1$", "$x^{*}_1$", "$R_{x1}$", ""],
                                               ["$<c>$", "$<c>$", "$c^*$", "$R_c$", "$\\tau_c$"]):

        for c_not_x1, key, label in zip([0, 1], [x1key, ckey], [x1label, clabel]):
            if key != None:
                plt.subplot(5,2, 2*i - 1 + c_not_x1)
                inds = argsort(stan_data["redshifts"])
                plt.fill_between(stan_data["redshifts"][inds],
                                 scoreatpercentile(fit_params[key], 15.8655, axis = 0)[inds],
                                 scoreatpercentile(fit_params[key], 84.1345, axis = 0)[inds])

                if i < 3:
                    zbins = linspace(stan_data["redshifts"].min()*0.999, stan_data["redshifts"].max()*1.001, 12)
                    for j in range(len(zbins) - 1):
                        inds = where((stan_data["redshifts"] >= zbins[j]) & (stan_data["redshifts"] < zbins[j+1]) & (isoutl < 0))
                        plt.plot(mean(stan_data["redshifts"][inds]), mean(array(stan_data["obs_mBx1c"])[:,1+c_not_x1][inds]), 'o', color = 'k', label = "Binned"*(j == 0))

                    ylim = plt.ylim()
                    plt.plot(stan_data["redshifts"], array(stan_data["obs_mBx1c"])[:,1+c_not_x1], '.', color = 'lightgray')
                    inds = where(isoutl > 0)
                    plt.plot(stan_data["redshifts"][inds], array(stan_data["obs_mBx1c"])[:,1+c_not_x1][inds], 'o', color = 'cyan', label = "Outliers")



                    plt.legend(loc = 'best')
                    
                    if i > 1:
                        #plt.ylim(ylim)
                        pass
                    else:
                        inds = where((isoutl < 0)*(array(stan_data["obs_mBx1c"])[:,1+c_not_x1] > c_not_x1*0.3 + (1 - c_not_x1)*2))
                        for ind in inds[0]:
                            plt.text(stan_data["redshifts"][ind], stan_data["obs_mBx1c"][ind,1+c_not_x1], the_data["snpaths"][ind].split("/")[-1], size = 4)

                plt.ylabel(label)
                if log_scale:
                    plt.xscale('log')

                
    plt.xlabel("Redshift")
    plt.savefig(resdir + "x1color_pop_model" + "_log"*log_scale + ".pdf", bbox_inches = 'tight')
    plt.close()


def cluster_chi2(P, passdata):
    orig_redshifts = passdata[0]
    assert len(orig_redshifts) > 1

    dz = 0.001
    chi2 = sum(  ((P - orig_redshifts)/dz)**2.  )
    for i in range(len(orig_redshifts)):
        for j in range(i+1, len(orig_redshifts)):
            chi2 += 1./abs(P[i] - P[j] + 0.000000001)
    return chi2


def spread_cluster_SNe(orig_redshifts):
    if orig_redshifts.max() > 2 and len(orig_redshifts) > 1:
        [P, NA, NA] = miniNM_new(ministart = orig_redshifts, miniscale = [0.001]*len(orig_redshifts), chi2fn = cluster_chi2, passdata = orig_redshifts, verbose = False)
        return P
    else:
        return orig_redshifts


def make_Hubble_diagram(use_obs_color):
    if use_obs_color:
        color_term = stan_data["obs_mBx1c"][:,2]*median(fit_params["beta_B"])*(stan_data["obs_mBx1c"][:,2] < 0)
        color_term += stan_data["obs_mBx1c"][:,2]*median(fit_params["beta_R"])*(stan_data["obs_mBx1c"][:,2] > 0)
    else:
        color_term = median(fit_params["true_cB"], axis = 0)*median(fit_params["beta_B"]) + median(fit_params["true_cR"], axis = 0)*median(fit_params["beta_R"])
        
    
    mus = (stan_data["obs_mBx1c"][:,0]
           + median(fit_params["alpha"])*stan_data["obs_mBx1c"][:,1]
           - color_term
           - median(fit_params["MB"])
           )

    zlabels = array([0.17, 0.35, 0.95, 1.65])
    mulabels = 5*log10((1. + zlabels)*(1.00875*zlabels - 0.271648*zlabels**2. + 0.0340072*zlabels**3. + 0.000441432*zlabels**4.)) + 42.
    mustep = 0.5

    print("Hacking mu_plot!!!!"*20)

    z_plot = exp(linspace(log(0.01), log(2.25), 200))
    assert max(stan_data["redshifts"] < 2.25)

    mu_plot = 5*log10((1. + z_plot)*(1.00875*z_plot - 0.271648*z_plot**2. + 0.0340072*z_plot**3. + 0.000441432*z_plot**4.))

    Moffset = median(mus - 5*log10((1. + stan_data["redshifts"])*(1.00875*stan_data["redshifts"] - 0.271648*stan_data["redshifts"]**2. + 0.0340072*stan_data["redshifts"]**3. + 0.000441432*stan_data["redshifts"]**4.)))
    mu_plot += Moffset

    
    print(fit_params["model_mBx1c_cov"].shape)
    med_model_mBx1c_cov = median(fit_params["model_mBx1c_cov"], axis = 0)
    
    one_alpha_negbeta = [1., median(fit_params["alpha"]), -0.5*median(fit_params["beta_B"]) -0.5*median(fit_params["beta_R_low"])]
    print("one_alpha_negbeta", one_alpha_negbeta)
    dmus = sqrt(array([dot(one_alpha_negbeta, dot(med_model_mBx1c_cov[i], one_alpha_negbeta))
                  for i in range(stan_data["n_sne"])
                  ]))
    print(dmus)
                  

    print("No host-mass correction!!!")

    f = open(os.environ["UNITY"] + "/paramfiles/sample_names_colors.txt", 'r')
    lines = f.read()
    f.close()

    for i in range(20):
        lines = lines.replace("\t\t", "\t").replace(" \t", "\t").replace("\t ", "\t").replace("  ", " ")
    
    lines = lines.split('\n')

    for i in range(len(lines))[::-1]:
        if lines[i].count('\t') == 2:
            lines[i] = lines[i].split('\t')
        else:
            if len(lines[i].split(None)) > 1:
                print("Skipping ", lines[i])
            del lines[i]


    f = open(resdir + "HR_" + "obs_color"*use_obs_color + "true_color"*(1 - use_obs_color) + ".txt", 'w')
    f.write("#Sample\tHR\tdmu\tIsOutl\n")

    for i in range(stan_data["n_samples"]):
        inds = where((stan_data["sample_list"] == i + 1)&(isoutl < 0))
        zmean = max(stan_data["redshifts"][inds])
        samp_cat = argmin(abs(zmean - zlabels))

        ind = [item[0] for item in lines].index(the_data["sample_names"][i].split("/")[-1])

        plt.text(zlabels[samp_cat], mulabels[samp_cat], lines[ind][1].replace("_", "\n"), color = eval(lines[ind][2]), size = 8, va = 'top')
        mulabels[samp_cat] -= mustep*(1 + lines[ind][1].count("_"))

        plt_redshifts = spread_cluster_SNe(stan_data["redshifts"][inds])

        for j, plt_redshift in zip(inds[0], plt_redshifts):
            plt.plot([plt_redshift]*2, [mus[j] - dmus[j], mus[j] + dmus[j]], color = eval(lines[ind][2]), linewidth = 0.75)
            plt.plot(plt_redshift, mus[j], '.', color = eval(lines[ind][2]), markersize = 2.5)

            the_data["sample_names"]
            the_data["snpaths"]
            mus
            fit_params["model_mu"]

            towrite = [the_data["sample_names"][i].split("/")[-1], the_data["snpaths"][j].split("/")[-1], mus[j] - median(fit_params["model_mu"][:,j]), dmus[j], isoutl[j]]
            f.write('\t'.join([str(item) for item in towrite]) + '\n')

    f.close()

    print("z_plot, mu_plot", z_plot, mu_plot)
    plt.plot(z_plot, mu_plot, color = 'k', zorder = 0)
    plt.xlabel("Redshift")
    plt.xlim(0, 2.25)
    assert max(stan_data["redshifts"] < 2.25)

    plt.ylabel("Distance Modulus")
    plt.savefig(resdir + "Hubble_diagram_" + "obs_color"*use_obs_color + "true_color"*(1 - use_obs_color) + ".pdf", bbox_inches = 'tight')
    plt.close()

    plt.figure(figsize = (10, 10))
    HR_x1_c = [mus - median(fit_params["model_mu"], axis = 0), stan_data["obs_mBx1c"][:,1], stan_data["obs_mBx1c"][:,2]]
    HR_x1_c_names = ["Hubble Residual", "$x_1$", "$c$"]

    for i in range(3):
        for j in range(i+1, 3):
            plt.subplot(3, 3, i+j*3+1)
            plt.scatter(HR_x1_c[i], HR_x1_c[j], c = [percentileofscore(isoutl, item) for item in isoutl], s = array(isoutl > 0, dtype=float64) + 0.5, cmap = 'brg')
            plt.xlabel(HR_x1_c_names[i])
            plt.ylabel(HR_x1_c_names[j])
    
    plt.savefig(resdir + "outliers_corner_" + "obs_color"*use_obs_color + "true_color"*(1 - use_obs_color) + ".pdf", bbox_inches = 'tight')
    plt.close()


def unc_labeling(labels_indiv):
    new_labels = []

    for label_indiv in labels_indiv:
        if label_indiv.startswith("calibs_('Fundamental"):
            new_labels.append("Fundamental Calibration")
            
        elif label_indiv.startswith("calibs_MWEBV_"):
            new_labels.append("Milky Way Extinction")

        elif label_indiv.startswith("MB"):
            new_labels.append("Absolute Magnitude $M_B$")
            
        elif label_indiv == "calibs_IG_extinction":
            new_labels.append("Intergalactic Dust")

        elif label_indiv == "calibs_corr_redshift_sys":
            new_labels.append("Redshift Calibration")
            
        elif label_indiv == "calibs_electron_scattering":
            new_labels.append("Electron Scattering")
            
        elif label_indiv.startswith("calibs_BULK_"):
            new_labels.append("Bulk Flow")

        elif label_indiv.startswith("calibs_('Zeropoint'"):
            new_labels.append("Instrument Zeropoints")

        elif label_indiv.startswith("calibs_('Lambda'"):
            new_labels.append("Instrument Bandpasses")
            
        elif label_indiv.startswith("tau_c_") or label_indiv.startswith("R_c_") or label_indiv.startswith("c_star_"):
            new_labels.append("Color Population")

        elif label_indiv.startswith("tau_x1_") or label_indiv.startswith("R_x1_") or label_indiv.startswith("x1_star_"):
            new_labels.append("$x_1$ Population")

        elif label_indiv.startswith("mBx1c_int_variance"):
            new_labels.append("Unexplained Scatter")
            
        elif label_indiv.startswith("mobs_cut"):
            new_labels.append("Selection Effects")
        elif label_indiv == "beta_B":
            new_labels.append("$\\beta_B$")
        elif label_indiv == "beta_R":
            new_labels.append("$\\beta_R$")
        elif label_indiv == "alpha":
            new_labels.append("$\\alpha$")


        elif label_indiv == "delta_beta_R":
            new_labels.append("$\Delta \\beta_R$")
        elif label_indiv == "delta_h":
            new_labels.append("$\delta(z = \infty)$")
        elif label_indiv == "delta_0":
            new_labels.append("$\delta(z = 0)$")
            
        else:
            new_labels.append(label_indiv)
            
    return new_labels


def unc_analysis(explain, keys):
    labels_indiv = []
    explained_indiv = []

    labels = []
    explained = []

    for key in keys:
        if key in fit_params:
            sh = fit_params[key].shape

            total_explained_squared = 0.

            if len(sh) == 2:
                for j in range(sh[1]):
                    cmat = cov(fit_params[key][:,j], fit_params[explain])
                    explained_indiv.append(cmat[0,1]/sqrt(cmat[0,0]))
                    labels_indiv.append(get_label(key, sh, j=j))
                    total_explained_squared += explained_indiv[-1]**2.


            elif len(sh) == 3:
                for j in range(sh[1]):
                    for k in range(sh[2]):
                        cmat = cov(fit_params[key][:,j,k], fit_params[explain])
                        explained_indiv.append(cmat[0,1]/sqrt(cmat[0,0]))
                        labels_indiv.append(get_label(key, sh, j=j,k=k))
                        total_explained_squared += explained_indiv[-1]**2.

            else:
                cmat = cov(fit_params[key], fit_params[explain])
                explained_indiv.append(cmat[0,1]/sqrt(cmat[0,0]))
                labels_indiv.append(get_label(key, sh))
                total_explained_squared += explained_indiv[-1]**2.

            labels.append(key)
            explained.append(sqrt(total_explained_squared))

    new_labels = unc_labeling(labels_indiv)
    assert len(new_labels) == len(explained_indiv), str(len(new_labels)) + " " + str(len(explained_indiv))
    unique_new = list(set(new_labels))
    explained_new = np.zeros(len(unique_new), dtype=np.float64)

    
    for i in range(len(new_labels)):
        ind = unique_new.index(new_labels[i])
        explained_new[ind] += (explained_indiv[i])**2.
    explained_new = np.sqrt(explained_new)
            
    for expl, lbls in [(explained_indiv, labels_indiv), (explained, labels), (explained_new, unique_new)]:
        expl = abs(array(expl))
        lbls = array(lbls)
        inds = argsort(expl)[::-1]
        
        expl = expl[inds]
        lbls = lbls[inds]
        
        
        for i, item in enumerate(zip(expl, lbls[:50])):
            print("%.4f\t&\t%.3f\t&\t%.3f\t&\t%s \\\\" % (item[0],
                                                          item[0]**2./dot(expl, expl),
                                                          dot(expl[:i+1], expl[:i+1])/dot(expl, expl),
                                                          item[1]))
        
        print("Total expl:", sqrt(dot(expl, expl)), "of", std(fit_params[explain]))
    

def get_label_dict():
    label_dict = {}
    dataset_names = [the_data["sample_names"][i].split("/")[-1].replace("_v1.txt", "") for i in range(stan_data["n_samples"])]
    
    for key in ["mobs_cuts", "mobs_cut_sigmas", "MB"]:
        label_dict[key] = dataset_names

    label_dict["calibs"] = the_data["calib_names"]


    for key in ["c_star", "tau_c", "R_c", "x1_star"]:
        label_dict[key] = []

        for i in range(stan_data["n_samples"]):
            label_dict[key].append([])
                
            for j in range(stan_data["n_x1c_star"]):
                label_dict[key][i].append(dataset_names[i] + "_" + str(j))

    return label_dict


def plot_sample_mag_limits():
    plt.figure(figsize = (6,0.5*stan_data["n_samples"]))

    for i in range(stan_data["n_samples"]):
        if stan_data["n_samples"] > 1:
            mobs_165084 = scoreatpercentile(fit_params["mobs_cuts"][:,i], [15.8655, 50, 84.1345])
        else:
            mobs_165084 = scoreatpercentile(fit_params["mobs_cuts"], [15.8655, 50., 84.1345])
        plt.plot(mobs_165084, [i + 0.2]*3, color = 'k', label = (i == 0)*"Posterior")
        plt.text(mean(mobs_165084[::2]),
                      i - 0.2, label_dict["mobs_cuts"][i], ha = 'center')
        plt.plot(mobs_165084[1], i + 0.2, '.', color = 'k')
            
        plt.plot(the_data["est_mobs_cuts"][i], i + 0.4, '.', color = 'r')
        plt.plot([the_data["est_mobs_cuts"][i] - 0.5, the_data["est_mobs_cuts"][i] + 0.5], [i + 0.4]*2, color = 'r', label = (i == 0)*"Prior")
        print("PRIOR WIDTH HACK!!!!"*100)

    plt.ylim(-0.5, stan_data["n_samples"] - 0.5)
    plt.yticks([])

    xticks, NA = plt.xticks()
    for xtick in xticks:
        plt.axvline(xtick, color = 'k', linestyle = ":")
    
    plt.legend(loc = 'best')
    plt.xlabel("Limiting Mag (Observer-Frame)")
    plt.savefig(resdir + "Mag_limits.pdf", bbox_inches = 'tight')
    plt.close()


def count_outliers():
    if "outl_loglike_by_SN" not in fit_params:
        print("Can't count oultiers!")
        return None

    isoutl = median(fit_params["outl_loglike_by_SN"], axis = 0) - median(fit_params["inl_loglike_by_SN"], axis = 0)

    print("Total outliers:", sum(isoutl > 0), "of", len(isoutl))
    for i in range(stan_data["n_samples"]):
        inds = where(stan_data["sample_list"] == i + 1)

        print(label_dict["mobs_cuts"][i], "outliers:", sum(isoutl[inds] > 0), "of", len(inds[0]), end=' ')
        for ind in inds[0]:
            if isoutl[ind] > 0:
                print(the_data["snpaths"][ind].split("/")[-1], end=' ')
        print()
    return isoutl

    

def diagnositics_plot():
    plt.figure(figsize = (4, stan_data["n_samples"]*0.5))

    for i in range(stan_data["n_samples"]):
        sigint = fit_params["sigma_int"][:,i]
        
        sigint_cred = scoreatpercentile(sigint, [15.8655, 50., 84.1345])
        
        plt.plot(sigint_cred[1], i + 0.5, '.', color = 'k')
        plt.plot(sigint_cred[::2], [i + 0.5]*2, color = 'k')
        plt.text(sigint_cred[1], i + 0.2, label_dict["mobs_cuts"][i])
        print("sig_int", label_dict["mobs_cuts"][i], sigint_cred)
    plt.savefig(resdir + "diagnostics.pdf", bbox_inches = 'tight')
    plt.close()

def plot_dz():
    plt.figure(figsize = (4, stan_data["n_photoz"]*2))
    for i in range(stan_data["n_photoz"]):
        plt.hist(stan_data["dz"][:,i], bins = 20)
    plt.savefig(resdir + "dzs.pdf", bbox_inches = 'tight')
    plt.close()

    
input_fl, sample_fl = sys.argv[1:]

resdir = sample_fl.replace(".pickle", "").replace("samples", "results") + "/"
assert resdir != sample_fl

print("results dir ", resdir)

subprocess.getoutput("mkdir " + resdir)
#commands.getoutput("rm -f " + resdir + "*")

print("Reading at ", time.asctime())
try:
    fit_params = pickle.load(open(sample_fl, 'rb'))
except:
    fit_params = pickle.load(gzip.open(sample_fl, 'rb'))

for key in fit_params:
    print("fit_params", key, fit_params[key].shape)
    
try:
    (the_data, stan_data, params) = pickle.load(open(input_fl, 'rb'))
except:
    (the_data, stan_data, params) = pickle.load(gzip.open(input_fl, 'rb'))

print("Done!", time.asctime())

fit_params["beta_R"] = 0.5*(fit_params["beta_R_low"] + fit_params["beta_R_high"])
fit_params["delta_beta_R"] = fit_params["beta_R_high"] - fit_params["beta_R_low"]


for key in ["obs_mBx1c"]:
    stan_data[key] = array(stan_data[key])

label_dict = get_label_dict()
for key in label_dict:
    print("label_dict ", key, "->", label_dict[key])

isoutl = count_outliers()

make_mB_vs_z()
show_x1color_pop(log_scale = 0)
show_x1color_pop(log_scale = 1)

plot_sample_mag_limits()

make_Hubble_diagram(0)
make_Hubble_diagram(1)



if len(fit_params["MB"]) == 1:
    fit_params["MB-delta_0"] = fit_params["MB"] - fit_params["delta_0"]
else:
    fit_params["MB-delta_0"] = fit_params["MB"]*1.
    
    for i in range(len(fit_params["MB"][0])):
        fit_params["MB-delta_0"][:,i] -= fit_params["delta_0"]
        


unc_analysis("Om", ["MB", "alpha", "beta_B", "beta_R", "delta_beta_R", "delta_0", "delta_h", "mobs_cuts", "mobs_cut_sigmas", "c_star", "R_c", "tau_c", "calibs", "x1_star", "mBx1c_int_variance"])


make_calib_corner(["MWEBV_multnorm", "MWEBV_addnorm"], "MWEBV_corner.pdf")


make_corner(["Om", "alpha", "beta_B", "beta_R", "delta_beta_R", "MB", "delta_0", "delta_h", "outl_frac"], "Om_coeffs.pdf")
make_corner(["delta_beta_R", "delta_0", "delta_h"], "host_coeffs.pdf")

try:
    import kde_corner
    run_kde = 1
except:
    run_kde = 0

if run_kde:
    make_kde_corner(["delta_beta_R", "delta_0"], "host_coeffs_kde.pdf")
    
make_corner(["delta_beta_R", "delta_0", "delta_h"], "host_coeffs.pdf")
make_corner(["Om", "alpha", "beta_B", "beta_R", "delta_beta_R", "MB-delta_0", "delta_0", "delta_h", "outl_frac"], "Om_MBhigh_coeffs.pdf")
make_corner(["Om", "mobs_cuts", "mobs_cut_sigmas"], "Om_mB_cut.pdf")
make_corner(["Om", "mobs_cuts", "mobs_cut_sigmas", "MB", "beta_B", "beta_R", "delta_betaR"], "Om_mB_cut_beta.pdf")
make_corner(["Om", "mobs_cuts", "mobs_cut_sigmas", "c_star", "R_c", "tau_c"], "Om_mB_cut_cpop.pdf")


diagnositics_plot()


for i in range(5):
    subprocess.getoutput("printf \\a")
