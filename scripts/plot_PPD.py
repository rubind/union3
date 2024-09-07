import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import gzip
import sys
from scipy.stats import percentileofscore, scoreatpercentile, gamma, poisson
import tqdm
import time
from helper_functions import bin_samples_in_redshift

def do_bin(xvals, yvals):
    bins = np.linspace(xvals.min(), xvals.max(), 10)

    bin_x = []
    bin_y = []

    for i in range(len(bins) - 1):
        inds = np.where((xvals > bins[i])*(xvals <= bins[i+1]))
        
        bin_x.append(np.median(xvals[inds]))
        bin_y.append(np.median(yvals[inds]))
    return bin_x, bin_y

def bin_into_panels(all_processed_data):
    plt.figure(figsize = (9, 15))
    
    for mBx1c_ind in [0, 1, 2]:
        label = ["Magnitude", "Light-Curve Shape", "Color"][mBx1c_ind]
        x1cname = ["mB", "x1", "c"][mBx1c_ind]

        z_dataset = []
        for dataset_match_ind in np.unique(all_processed_data[x1cname]["per_dataset_ind"]):
            inds = np.where(all_processed_data[x1cname]["per_dataset_ind"] == dataset_match_ind)
            if len(inds[0]) >= 30:
                z_dataset.append((np.mean(all_processed_data[x1cname]["per_z"][inds]), dataset_match_ind))

        z_dataset.sort()
        z_dataset = z_dataset[::-1]

        for NA, dataset_match_ind in z_dataset:
            median_sample_z = -1 # show all SNe rather than above median np.median(per_z[np.where(per_dataset_ind == dataset_match_ind)])

            inds = np.where((per_dataset_ind == dataset_match_ind)*(all_processed_data[x1cname]["per_z"] >= median_sample_z))
            set_color = samp_bins["colors"][samp_bins["inds"].index(dataset_match_ind - 1)]

            plt.subplot(len(z_dataset), 3, tmp_ind*3 + mBx1c_ind - 2)
            PPD1D = (all_processed_data[x1cname]["tmp_c_PPD_found"][inds]).flatten()
            PPD1D_found_or_not = (all_processed_data[x1cname]["tmp_c_PPD_found_or_not"][inds]).flatten()

            if mBx1c_ind == 0: # If mB
                tmp_c_obs[inds] -= np.mean(PPD1D)
                PPD1D_found_or_not -= np.mean(PPD1D)
                PPD1D -= np.mean(PPD1D)

            #plt.hist(PPD1D_found_or_not, bins = bins, label = "PPD", density=True, color = 'b', histtype = "step")
            plt.hist(PPD1D, bins = bins, label = "PPD", density=True, color = 'k', histtype = "step")
            #plt.title(the_data["sample_names"][dataset_match_ind - 1].split("/")[-1].split("_v1")[0])
            #plt.title(samp_bins["short_labels"][samp_bins["inds"].index(dataset_match_ind - 1)])# + the_data["sample_names"][dataset_match_ind - 1].split("/")[-1].split("_v1")[0])

            the_norm = 1./((bins[1] - bins[0])*(len(inds[0])))
            the_gain = len(inds[0])/len(PPD1D)

            chi2s = []

            for i in range(len(bins) - 1):
                the_count = sum((per_dataset_ind == dataset_match_ind)*(tmp_c_obs >= bins[i])*(tmp_c_obs < bins[i+1])*(per_z >= median_sample_z))
                the_count_PPD = sum((PPD1D >= bins[i])*(PPD1D < bins[i+1])) * the_gain

                """
                alpha_prior = 1  # Shape parameter for the prior (uniform prior)
                interval_confidence = 0.683  # For a ~68% credible interval


                alpha_post = alpha_prior + the_count
                beta_post = 1  # Since we assume a uniform prior

                # Lower and upper bounds of the credible interval
                lower_bound = gamma.ppf((1 - interval_confidence) / 2, alpha_post, scale=beta_post)
                upper_bound = gamma.ppf((1 + interval_confidence) / 2, alpha_post, scale=beta_post)
                """

                plt.plot(0.5*(bins[i] + bins[i+1]),
                             the_count*the_norm, '.', color = set_color,
                         label = "REPLACE"*(i == 0), zorder = 5)

                lower_bound = poisson.ppf(0.158655, the_count_PPD)
                upper_bound = poisson.ppf(0.841345, the_count_PPD)

                #print("the_count_PPD", the_count_PPD, lower_bound, upper_bound)

                plt.plot([0.5*(bins[i] + bins[i+1])]*2,
                         [lower_bound*the_norm, upper_bound*the_norm], color = 'k', zorder = 0)

                if the_count_PPD > 0:
                    chi2s.append((the_count - the_count_PPD)**2./the_count_PPD)

                plt.yticks([])


            legend = plt.legend(loc = 'upper right', fontsize = 8, bbox_to_anchor=(1.4, 1.05))
            texts = legend.get_texts()
            texts[1].set_text("$\chi^2/N_{\mathrm{bins}}$\n= %.1f/%i" % (sum(chi2s), len(chi2s))) # texts[-1].get_text() + 

            plt.xlim(bins[0], bins[-1])
            if tmp_ind < len(z_dataset):
                plt.xticks(visible=False)
            plt.xlim(bins[0], bins[-1])

            if tmp_ind == 1:
                plt.title(label + [" $m_B - $ Sample Mean", " $x_1$", " $c$"][mBx1c_ind], fontsize = 12)
            if mBx1c_ind == 0:
                plt.ylabel(samp_bins["short_labels"][samp_bins["inds"].index(dataset_match_ind - 1)].replace("(", "\n("), fontsize = 9)

            tmp_ind += 1
        plt.xlabel(label + [" $m_B - $ Sample Mean", " $x_1$", " $c$"][mBx1c_ind], fontsize = 12)


    plt.tight_layout(h_pad = 0.3)
    plt.savefig("PPD_by_sample.pdf", bbox_inches = 'tight')
    plt.close()



input_fl = sys.argv[1]

(the_data, stan_data, params) = pickle.load(gzip.open(input_fl, 'rb'))

for key in the_data:
    print("the_data", key)

samp_bins = bin_samples_in_redshift(stan_data, the_data)
    
for key in stan_data:
    print("stan_data", key)

for key in samp_bins:
    print("samp_bins", key)

print('samp_bins["inds"]', samp_bins["inds"])

assert np.min(stan_data["sample_list"]) == 1
assert len(the_data["sample_names"]) == np.max(stan_data["sample_list"])

    
print("Loading PPD...", time.asctime())
PPD = pickle.load(open("PPD.pickle", 'rb'))
print("Done", time.asctime())

for key in PPD:
    print("PPD", key) # Can't show shape, as size will be zero if SN is an outlier, so PPD[key] is ragged

"""
PPD obs_mBx1c
PPD found
PPD true_mBx1cBcR
PPD obs_mBx1c|foundobsx1c
PPD true_mBx1cBcR|foundobsx1c
PPD obs_mBx1c|obsx1c
PPD true_mBx1cBcR|obsx1c
"""



for SN in tqdm.trange(len(PPD["obs_mBx1c"])):
    if len(PPD["obs_mBx1c"][SN]) > 0:
        PPD["obs_mBx1c"][SN] = np.array(PPD["obs_mBx1c"][SN])
        
        inds = np.where(PPD["found"][SN])

        mu_obs = PPD["obs_mBx1c"][SN][:, 0] + 0.14*PPD["obs_mBx1c"][SN][:, 1] - 3.1*PPD["obs_mBx1c"][SN][:, 2]

        mean_mu = np.mean(mu_obs)
        mean_found = np.mean(mu_obs[inds])

        #plt.errorbar(stan_data["redshifts"][SN], mean_found - mean_mu, yerr = np.std(mu_obs[inds], ddof = 1)/np.sqrt(float(len(inds[0]))), fmt = '.', color = 'b')


        plt.plot(stan_data["redshifts"][SN], mean_found - mean_mu, '.', color = 'b')

    #if mean_found - mean_mu > 0.2:
    #    plt.text(stan_data["redshifts"][SN])
plt.savefig("delta_mu_vs_z.pdf")
plt.close()

plt.figure(3, figsize = (9, 15))

plt.figure(4, figsize = (10, 6))


all_processed_data = {}

for mBx1c_ind in [0, 1, 2]:
    label = ["Magnitude", "Light-Curve Shape", "Color"][mBx1c_ind]
    second_label = ["\n$\leftarrow$ Fainter          Brighter $\\rightarrow$", "\n$\leftarrow$ Lower $x_1$          Higher $x_1$ $\\rightarrow$", "\n$\leftarrow$ Bluer          Redder $\\rightarrow$"][mBx1c_ind]
    x1cname = ["mB", "x1", "c"][mBx1c_ind]


    all_processed_data[x1cname] = dict(
        per_z = [], # percentiles
        per_per = [],
        per_sel = [],
        per_dataset_ind = [],
        
        tmp_c_obs = [],
        tmp_c_PPD_found = [],
        tmp_c_PPD_found_or_not = [],
        tmp_c_PPD_RMS = [],
        median_dc = [])

    for SN in tqdm.trange(len(PPD["obs_mBx1c"])):
        if len(PPD["obs_mBx1c"][SN]) > 0:
            inds = np.where(PPD["found"][SN])
            c_found = PPD["obs_mBx1c"][SN][:, mBx1c_ind][inds]
            c_found_or_not = PPD["obs_mBx1c"][SN][:, mBx1c_ind]
            c_obs = stan_data["obs_mBx1c"][SN][mBx1c_ind]

            all_processed_data[x1cname]["median_dc"].append(np.median(PPD[x1cname + "_unc"][SN]))
                            
            all_processed_data[x1cname]["tmp_c_obs"].append(c_obs)
            all_processed_data[x1cname]["tmp_c_PPD_found"].append(np.random.choice(c_found, size = 100, replace = True))
            all_processed_data[x1cname]["tmp_c_PPD_found_or_not"].append(np.random.choice(c_found_or_not, size = 100, replace = True))
            all_processed_data[x1cname]["tmp_c_PPD_RMS"].append(np.std(c_found, ddof=1))
            
            per = percentileofscore(c_found, c_obs)
            all_processed_data[x1cname]["per_z"].append(stan_data["redshifts"][SN])
            all_processed_data[x1cname]["per_per"].append(per)
            all_processed_data[x1cname]["per_sel"].append(np.mean(PPD["found"][SN]))
            all_processed_data[x1cname]["per_dataset_ind"].append(stan_data["sample_list"][SN])

    for key in all_processed_data[x1cname]:
        all_processed_data[x1cname][key] = np.array(all_processed_data[x1cname][key])

        
    plt.figure(1, figsize = (15, 5))
    plt.subplot(1,3,1)
    best_measured_half_inds = np.where(all_processed_data[x1cname]["median_dc"] < np.median(all_processed_data[x1cname]["median_dc"]))

    plt.hist(all_processed_data[x1cname]["per_per"], bins = np.arange(0., 101., 5.), color = 'c', label = "Worst-Measured Half", orientation="horizontal")
    plt.hist(all_processed_data[x1cname]["per_per"][best_measured_half_inds], bins = np.arange(0., 101., 5.), color = 'b', label = "Best-Measured Half", orientation="horizontal")

    plt.legend(loc = 'best')
    
    plt.ylim(0, 100)
    plt.xlabel("Count")
    plt.ylabel("Percentile of Observed " + label + second_label)
    
    
    plt.subplot(1,3,2)
    plt.plot(all_processed_data[x1cname]["per_z"], all_processed_data[x1cname]["per_per"], '.', color = 'b')
    plt.xlabel("Redshift")
    plt.ylabel("Percentile of Observed " + label + second_label)
    plt.ylim(0, 100)
    plt.xscale('log')

    plt.subplot(1,3,3)
    plt.plot(all_processed_data[x1cname]["per_sel"], all_processed_data[x1cname]["per_per"], '.', color = 'b')
    plt.xlabel("Probability of Selection")
    plt.xlim(0, 1)
    plt.ylabel("Percentile of Observed " + label + second_label)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(label.replace(" ", "_") + "_percentile.pdf", bbox_inches = 'tight')
    plt.close()


    
    bins = np.linspace(min(np.min(all_processed_data[x1cname]["tmp_c_obs"]), np.min(all_processed_data[x1cname]["tmp_c_PPD_found"])),
                       max(np.max(all_processed_data[x1cname]["tmp_c_obs"]), np.max(all_processed_data[x1cname]["tmp_c_PPD_found"])), 41)

    
    
    plt.figure(2)
    
    plt.hist(all_processed_data[x1cname]["tmp_c_obs"], bins = bins, alpha = 0.5, label = "Observed", density = True)
    plt.hist(all_processed_data[x1cname]["tmp_c_PPD_found"].flatten(), bins = bins, alpha = 0.5, label = "PPD", density=True)
    # each SN gets same normalization because we pull 100 samples for each
    plt.legend(loc = 'best')
    
    plt.savefig(label.replace(" ", "_") + "_histogram.pdf", bbox_inches = 'tight')
    plt.close()


    tmp_ind = 1
    #bins = np.linspace(min(np.min(tmp_c_obs), np.min(tmp_c_PPD_found)),
    #                   max(np.max(tmp_c_obs), np.max(tmp_c_PPD_found)), 21)

    if mBx1c_ind == 2:
        bins = np.linspace(-0.35, 0.35, 15)
    elif mBx1c_ind == 1:
        bins = np.linspace(-4, 4, 15)
    else:
        bins = np.linspace(-3, 3, 15)

        
    
    plt.figure(4)
    
    for mean_not_median in [0, 1]*(mBx1c_ind > 0):
        c_not_x1 = mBx1c_ind - 1
        plt.subplot(2,2, c_not_x1*2 +  2 - mean_not_median)
        
        all_ppdz_to_plot = []
        all_ppd_to_plot = []
        all_ppd_unc_to_plot = []
        mean_median_word = "Mean"*mean_not_median + "Median"*(1 - mean_not_median)
        
        for i in range(len(zbins) - 1):
            inds = np.where((all_processed_data[x1cname]["per_z"] >= zbins[i])*(all_processed_data[x1cname]["per_z"] < zbins[i+1]))
            rms = np.std(all_processed_data[x1cname]["tmp_c_obs"][inds], ddof=1)
            mean_PPD_unc = np.mean(all_processed_data[x1cname]["tmp_c_PPD_RMS"][inds])

            if mean_not_median:
                valz = np.mean(all_processed_data[x1cname]["per_z"][inds])
                valy = np.mean(all_processed_data[x1cname]["tmp_c_obs"][inds])
                uncmean = rms/np.sqrt(1.*len(inds[0]))

                all_ppdz_to_plot.append(np.mean(all_processed_data[x1cname]["per_z"][inds]))
                all_ppd_to_plot.append(np.mean(all_processed_data[x1cname]["tmp_c_PPD_found"][inds]))
                all_ppd_unc_to_plot.append(np.mean(all_processed_data[x1cname]["tmp_c_PPD_RMS"][inds]))

            else:
                valz = np.median(all_processed_data[x1cname]["per_z"][inds])
                valy = np.median(all_processed_data[x1cname]["tmp_c_obs"][inds])
                uncmean = np.sqrt(np.pi/2.)*rms/np.sqrt(1.*len(inds[0]))

                all_ppdz_to_plot.append(np.median(all_processed_data[x1cname]["per_z"][inds]))
                all_ppd_to_plot.append(np.median(all_processed_data[x1cname]["tmp_c_PPD_found"][inds]))
                all_ppd_unc_to_plot.append(np.sqrt(np.pi/2.)*np.median(all_processed_data[x1cname]["tmp_c_PPD_RMS"][inds]))

                
            plt.errorbar(valz,
                         valy,
                         yerr = uncmean,
                         fmt = '.', color = 'b', label = ("Observed " + mean_median_word)*(i == 0))

            
            plt.plot(all_ppdz_to_plot[-1],
                     all_ppd_to_plot[-1], '^', color = 'r', label = ("PPD " + mean_median_word)*(i == 0))

        
        all_ppd_to_plot = np.array(all_ppd_to_plot)
        all_ppd_unc_to_plot = np.array(all_ppd_unc_to_plot)
        
        #plt.fill_between(all_ppdz_to_plot, all_ppd_to_plot - all_ppd_unc_to_plot,
        #                 all_ppd_to_plot + all_ppd_unc_to_plot, color = 'r')

        plt.legend(loc = 'best')
        plt.xscale('log')
        plt.xlabel("Redshift")
        plt.ylabel(label + ", Equal-SN Bins")
        #plt.savefig("median"*(mean_not_median == 0) + "mean"*mean_not_median + "_" + x1cname + "_vs_z.pdf", bbox_inches = 'tight')
        #plt.close()
plt.figure(4)
plt.tight_layout()
plt.savefig("mean_and_median_vs_z.pdf", bbox_inches = 'tight')
plt.close()


    
plt.scatter(all_processed_data["c"]["per_z"], all_processed_data["c"]["per_sel"])
plt.xscale('log')
plt.xlabel("Redshift")
plt.ylabel("Probability of Selection")
plt.ylim(0, 1)
plt.savefig("prob_sel.pdf", bbox_inches = 'tight')
plt.close()



plt.figure(figsize = (8, 18))

for SN in tqdm.trange(len(PPD["obs_mBx1c"])):
    if len(PPD["obs_mBx1c"][SN]) > 0:
        PPD["obs_mBx1c|foundobsx1c"][SN] = np.array(PPD["obs_mBx1c|foundobsx1c"][SN])

        for i in range(3):
            plt.subplot(3,1,1+i)
            median_mB = np.median(PPD["obs_mBx1c|foundobsx1c"][SN][:, i])
            obs_mB = stan_data["obs_mBx1c"][SN][i]
            
            plt.plot(stan_data["redshifts"][SN], obs_mB - median_mB, '.', color = 'b')
            if np.abs(obs_mB - median_mB) > 0.7:
                plt.text(stan_data["redshifts"][SN], obs_mB - median_mB, the_data["snpaths"][SN].split("/")[-1], fontsize = 6)
            plt.title(["mB", "x1", "c"][i])

for i in range(3):
    plt.subplot(3,1,1+i)
    plt.axhline(0)
plt.savefig("Hubble_resid_PPD.pdf", bbox_inches = 'tight')
plt.close()



bin_into_panels(all_processed_data)



ffffff

fit_params = pickle.load(gzip.open(sys.argv[2], 'rb'))

print(PPD["obs_mBx1c"].shape)


for SN in range(len(PPD["obs_mBx1c"])):
    inds = np.where(PPD["found"][SN])

PPD_resid = PPD["obs_mBx1c"][:, :, 0] + 0.14*PPD["obs_mBx1c"][:, :, 1] - 3.1*PPD["obs_mBx1c"][:, :, 2] - fit_params["model_mu"].T - fit_params["MB"].T
print("PPD_resid", PPD_resid.shape)

obs_resid = stan_data["obs_mBx1c"][:, 0] + 0.14*stan_data["obs_mBx1c"][:, 1] - 3.1*stan_data["obs_mBx1c"][:, 2] - np.median(fit_params["model_mu"].T) - np.median(fit_params["MB"].T)
print("obs_resid", obs_resid.shape)
print("obs_resid", obs_resid)

bin_x, bin_y = do_bin(stan_data["obs_mBx1c"][:, 2], obs_resid)

plt.scatter(bin_x, bin_y, label = "Observed")
plt.xlabel("Color")
plt.ylabel("Linear Hubble Diagram Residual")
plt.savefig("standard_color.pdf", bbox_inches = 'tight')
plt.close()

