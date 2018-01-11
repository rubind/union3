from numpy import * # I know...
import cPickle as pickle
from matplotlib import use
use("PDF")
import corner # I know...
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
import sys
import commands
import gzip
from FileRead import readcol
import time
import os
plt.rcParams["font.family"] = "serif"


def get_label(key, sh, j=None, k=None):
    if len(sh) == 2:
        return key + "_" + str(label_dict.get(key, [j]*(j + 1))[j]
                               )
    elif len(sh) == 3:
        return key + "_" + str(label_dict.get(key, [[(j,k)]*(k + 1)]*(j + 1))[j][k]
        )
    else:
        return key


def make_corner(keys, pltname):
    print "Corner plot for ", keys
    samples = []
    labels = []

    for key in keys:
        if fit_params.has_key(key):
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
            print "Skipping ", key
            
    samples = transpose(array(samples))

    corner.corner(samples, labels = labels)
    plt.savefig(resdir + pltname)
    plt.close()


def make_mB_vs_z():
    plt.subplot(2,1,1)
    plt.plot(stan_data["redshifts"], array(stan_data["obs_mBx1c"])[:,0], '.', color = 'k')
    plt.subplot(2,1,2)
    plt.hist(array(stan_data["obs_mBx1c"])[:,0])
    plt.savefig(resdir + "mB_vs_z.pdf")
    plt.close()


def show_color_pop():
    plt.figure(figsize = (8,12))

    fit_params["mean_c_by_SN"] = fit_params["c_star_by_SN"] + fit_params["tau_c_by_SN"]

    for i, key, label in zip([1,2,3,4], ["mean_c_by_SN", "c_star_by_SN", "R_c_by_SN", "tau_c_by_SN"], ["$<c>$", "$c^*$", "$R_c$", "$\\tau_c$"]):
        plt.subplot(4,1,i)
        inds = argsort(stan_data["redshifts"])
        plt.fill_between(stan_data["redshifts"][inds],
                         scoreatpercentile(fit_params[key], 15.8655, axis = 0)[inds],
                         scoreatpercentile(fit_params[key], 84.1345, axis = 0)[inds])

        if i == 1:
            zbins = linspace(stan_data["redshifts"].min()*0.999, stan_data["redshifts"].max()*1.001, 12)
            for j in range(len(zbins) - 1):
                inds = where((stan_data["redshifts"] >= zbins[j]) & (stan_data["redshifts"] < zbins[j+1]))
                plt.plot(mean(stan_data["redshifts"][inds]), mean(array(stan_data["obs_mBx1c"])[:,2][inds]), 'o', color = 'k', label = "Binned"*(j == 0))
            ylim = plt.ylim()
            plt.plot(stan_data["redshifts"], array(stan_data["obs_mBx1c"])[:,2], '.', color = 'lightgray')
            plt.legend(loc = 'best')
            plt.ylim(ylim)
        plt.ylabel(label)
    plt.xlabel("Redshift")
    plt.savefig(resdir + "color_pop_model.pdf", bbox_inches = 'tight')
    plt.close()


def make_Hubble_diagram():
    color_term = median(fit_params["true_cB"], axis = 0)*median(fit_params["beta_B"]) + median(fit_params["true_cR"], axis = 0)*median(fit_params["beta_R"])
    #color_term *= stan_data["obs_mBx1c"][:,2]/(median(fit_params["true_cB"], axis = 0) + median(fit_params["true_cR"], axis = 0))
    
    
    mus = (stan_data["obs_mBx1c"][:,0]
           + median(fit_params["alpha"])*stan_data["obs_mBx1c"][:,1]
           - color_term
           - median(fit_params["MB"])
           )

    zlabels = array([0.1, 0.3, 0.6, 1.5])
    mulabels = 5*log10((1. + zlabels)*(1.00875*zlabels - 0.271648*zlabels**2. + 0.0340072*zlabels**3. + 0.000441432*zlabels**4.)) + 41.
    mustep = 0.55
    

    print fit_params["model_mBx1c_cov"].shape
    med_model_mBx1c_cov = median(fit_params["model_mBx1c_cov"], axis = 0)
    
    one_alpha_negbeta = [1., median(fit_params["alpha"]), -0.5*median(fit_params["beta_B"]) -0.5*median(fit_params["beta_R"])]
    print "one_alpha_negbeta", one_alpha_negbeta
    dmus = sqrt(array([dot(one_alpha_negbeta, dot(med_model_mBx1c_cov[i], one_alpha_negbeta))
                  for i in range(stan_data["n_sne"])
                  ]))
    print dmus
                  
    isoutl = median(fit_params["outl_loglike_by_SN"], axis = 0) - median(fit_params["inl_loglike_by_SN"], axis = 0)

    print "No host-mass correction!!!"

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
                print "Skipping ", lines[i]
            del lines[i]

    for i in range(stan_data["n_samples"]):
        inds = where((stan_data["sample_list"] == i + 1)&(isoutl < 0))
        zmean = mean(stan_data["redshifts"][inds])
        samp_cat = argmin(abs(zmean - zlabels))

        ind = [item[0] for item in lines].index(the_data["sample_names"][i].split("/")[-1])

        plt.text(zlabels[samp_cat], mulabels[samp_cat], lines[ind][1], color = eval(lines[ind][2]))
        mulabels[samp_cat] -= mustep

        for j in inds[0]:
            plt.plot([stan_data["redshifts"][j]]*2, [mus[j] - dmus[j], mus[j] + dmus[j]], color = eval(lines[ind][2]))
            plt.plot(stan_data["redshifts"][j], mus[j], '.', color = eval(lines[ind][2]))

    plt.xlabel("Redshift")
    plt.xlim(0, plt.xlim()[1])
    plt.ylabel("Distance Modulus")
    plt.savefig(resdir + "Hubble_diagram.pdf", bbox_inches = 'tight')
    plt.close()


def error_analysis(explain, keys):
    labels = []
    explained = []

    for key in keys:
        if fit_params.has_key(key):
            sh = fit_params[key].shape

            if len(sh) == 2:
                for j in range(sh[1]):
                    cmat = cov(fit_params[key][:,j], fit_params[explain])
                    explained.append(cmat[0,1]/sqrt(cmat[0,0]))
                    labels.append(get_label(key, sh, j=j))

            elif len(sh) == 3:
                for j in range(sh[1]):
                    for k in range(sh[2]):
                        cmat = cov(fit_params[key][:,j,k], fit_params[explain])
                        explained.append(cmat[0,1]/sqrt(cmat[0,0]))
                        labels.append(get_label(key, sh, j=j,k=k))

            else:
                cmat = cov(fit_params[key], fit_params[explain])
                explained.append(cmat[0,1]/sqrt(cmat[0,0]))
                labels.append(get_label(key, sh))

    explained = abs(array(explained))
    labels = array(labels)
    inds = argsort(explained)[::-1]
    
    explained = explained[inds]
    labels = labels[inds]
    

    for i, item in enumerate(zip(explained, labels[:50])):
        print "%.3g\t\t%.3g\t\t%.3g\t\t%s" % (item[0]**2./dot(explained, explained),
                                        dot(explained[:i+1], explained[:i+1])/dot(explained, explained),
                                        item[0], item[1])
        
    print "Total explained:", sqrt(dot(explained, explained)), "of", std(fit_params[explain])
    

def get_label_dict():
    label_dict = {}
    for key in ["mobs_cuts", "mobs_cut_sigmas"]:
        label_dict[key] = [the_data["sample_names"][i].split("/")[-1].replace("_v1.txt", "") for i in range(stan_data["n_samples"])]

    label_dict["calibs"] = the_data["calib_names"]

    for key in ["c_star", "tau_c", "R_c", "x1_star"]:
        label_dict[key] = []

        for i in range(stan_data["n_samples"]):
            label_dict[key].append([])
            for j in range(stan_data["n_x1c_star"]):
                label_dict[key][i].append(label_dict["mobs_cuts"][i] + "_" + str(j))

    return label_dict

def plot_sample_mag_limits():
    for i in range(stan_data["n_samples"]):
        if stan_data["n_samples"] > 1:
            mobs_165084 = scoreatpercentile(fit_params["mobs_cuts"][:,i], [15.8655, 50, 84.1345])
        else:
            mobs_165084 = scoreatpercentile(fit_params["mobs_cuts"], [15.8655, 50., 84.1345])
        plt.plot(mobs_165084, [i + 0.2]*3, color = 'k')
        plt.text(mean(mobs_165084[::2]),
                      i - 0.2, label_dict["mobs_cuts"][i], ha = 'center')
        plt.plot(mobs_165084[1], i + 0.2, '.', color = 'k')
            
        plt.plot(the_data["est_mobs_cuts"][i], i + 0.4, '.', color = 'r')
    plt.savefig(resdir + "Mag_limits.pdf", bbox_inches = 'tight')
    plt.close()


def count_outliers():
    if not fit_params.has_key("outl_loglike_by_SN"):
        print "Can't count oultiers!"
        return None

    isoutl = median(fit_params["outl_loglike_by_SN"], axis = 0) - median(fit_params["inl_loglike_by_SN"], axis = 0)

    print "Total outliers:", sum(isoutl > 0), "of", len(isoutl)
    for i in range(stan_data["n_samples"]):
        inds = where(stan_data["sample_list"] == i + 1)

        print label_dict["mobs_cuts"][i], "outliers:", sum(isoutl[inds] > 0), "of", len(inds[0]),
        for ind in inds[0]:
            if isoutl[ind] > 0:
                print the_data["snpaths"][ind].split("/")[-1],
        print

        
        
    
input_fl, sample_fl = sys.argv[1:]

resdir = sample_fl.replace(".pickle", "").replace("samples", "results") + "/"
assert resdir != sample_fl

print "results dir ", resdir

commands.getoutput("mkdir " + resdir)
#commands.getoutput("rm -f " + resdir + "*")

print "Reading at ", time.asctime()
fit_params = pickle.load(gzip.open(sample_fl, 'rb'))
(the_data, stan_data, params) = pickle.load(gzip.open(input_fl, 'rb'))
print "Done!", time.asctime()

for key in ["obs_mBx1c"]:
    stan_data[key] = array(stan_data[key])

label_dict = get_label_dict()
make_mB_vs_z()
show_color_pop()
make_Hubble_diagram()
plot_sample_mag_limits()
count_outliers()
fff

error_analysis("Om", ["MB", "alpha", "beta_B", "beta_R", "delta_0", "delta_h", "mobs_cuts", "mobs_cut_sigmas", "c_star", "R_c", "tau_c", "calibs", "x1_star"])


make_corner(["Om", "alpha", "beta_B", "beta_R", "delta_0", "delta_h", "outl_frac"], "Om_coeffs.pdf")
make_corner(["Om", "mobs_cuts", "mobs_cut_sigmas"], "Om_mB_cut.pdf")
make_corner(["Om", "mobs_cuts", "mobs_cut_sigmas", "MB", "beta_B", "beta_R"], "Om_mB_cut_beta.pdf")
make_corner(["Om", "mobs_cuts", "mobs_cut_sigmas", "c_star", "R_c", "tau_c"], "Om_mB_cut_cpop.pdf")


for i in range(5):
    commands.getoutput("printf \\a")
