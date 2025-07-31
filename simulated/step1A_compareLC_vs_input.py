from FileRead import readcol, read_param
import glob
import matplotlib.pyplot as plt
import numpy as np
from subprocess import getoutput
import tqdm
import sys
from scipy.stats import scoreatpercentile
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM
from DavidsNM import miniLM_new
import pickle


def do_med_bins(x, y, sigy, nbins, average_not_median = 0):
    bad_mask = np.isnan(x) + np.isnan(y) + np.isnan(sigy)
    
    inds = np.where(bad_mask == 0)
    binedges = scoreatpercentile(x[inds], np.linspace(0, 100, nbins+1))
    binedges[0] -= (binedges[1] - binedges[0])*0.001
    binedges[-1] += (binedges[-1] - binedges[-2])*0.001

    binx = []
    biny = []
    
    for i in range(nbins):
        inds = np.where((x >= binedges[i])*(x < binedges[i+1])*(bad_mask == 0))

        if average_not_median:
            if sum(1./sigy[inds]**2.) > 0:
                binx.append(sum(x[inds]/sigy[inds]**2.)/sum(1./sigy[inds]**2.))
                biny.append(sum(y[inds]/sigy[inds]**2.)/sum(1./sigy[inds]**2.))
            else:
                binx.append(np.sqrt(-1.))
                biny.append(np.sqrt(-1.))
        else:
            binx.append(np.median(x[inds]))
            biny.append(np.median(y[inds]))
    return np.array(binx), np.array(biny)

def get_dmu(bands, resfl):
    f = open(resfl.replace("result_salt2.dat", "result_deriv.dat"), 'r')
    grepout = f.read().split('\n')
    f.close()
    
    grepout = [line for line in grepout if line.count("Zeropoint") == 1]
    

    dmudzps = dict(g = 0, r = 0, i = 0, z = 0)
    
    for band in bands:
        for line in grepout:
            if line.count("SDSS_" + band):
                dmudzps[band] = float(line.split(None)[4])
    return dmudzps


def modelfn(P, passdata):
    [zs, delta_mus, mu_uncs] = passdata[0]

    cosmo = Flatw0waCDM(Om0 = P[1], H0 = 70., w0 = P[2], wa = P[3])
    cosmo3 = Flatw0waCDM(Om0 = 0.3, H0 = 70., w0 = -1, wa = 0)
    mu = cosmo.distmod(zs).value
    mu3 = cosmo3.distmod(zs).value
    return mu - mu3 + P[0]
    

def pullfn(P, passdata):
    [zs, delta_mus, mu_uncs] = passdata[0]
    delta_mu_model = modelfn(P, passdata)
    
    return (delta_mus - delta_mu_model)/mu_uncs


def fit_delta_cosmo(zs, delta_mus, mu_uncs, pltzs, fit_Om, fit_w0, fit_wa, verbose = False, n_boot = 50):
    P, NA, NA = miniLM_new(ministart = [0.0, 0.3, -1, 0.], miniscale = [1., fit_Om, fit_w0, fit_wa], residfn = pullfn, passdata = [zs, delta_mus, mu_uncs], verbose = verbose, maxiter = 3)
    # The mu_uncs have 0.12 added in quadrature, so the uncertainties are overestimated compared to just the LC fit uncertainties. Need to bootstrap.

    P_reals = []
    for i in tqdm.trange(n_boot):
        new_inds = np.random.choice(np.arange(len(zs), dtype=np.int32), size = len(zs), replace = True)

        P_tmp, NA, NA = miniLM_new(ministart = [0.0, 0.3, -1, 0.], miniscale = [1., fit_Om, fit_w0, fit_wa], residfn = pullfn, passdata = [zs[new_inds], delta_mus[new_inds], mu_uncs[new_inds]], verbose = verbose, maxiter = 3)
        P_reals.append(P_tmp)

    P_reals = np.array(P_reals).T

    if fit_Om == 0:
        the_label = "(Unbinned) Fit:\n$\Omega_m = %.3f (\mathrm{fixed})$" % (P[1])
    else:
        the_label = "(Unbinned) Fit:\n$\Omega_m = %.3f \pm %.3f$" % (P[1], np.std(P_reals[1], ddof=1))
        
    if fit_w0:
        the_label = the_label.replace("\n", " ") + ", $w_0 = %.3f \pm %.3f$,\n$w_0 + 0.15\;w_a= %.3f \pm %.3f$, $w_a = %.3f \pm %.3f$" % (P[2], np.std(P_reals[2], ddof=1),
                                                                                                                                          P[2] + 0.15*P[3], np.std(P_reals[2] + 0.15*P_reals[3], ddof=1),
                                                                                                                                          P[3], np.std(P_reals[3], ddof=1))
        
    return modelfn(P, [[pltzs, None, None]]), the_label


def read_dat():
    all_dat = dict(true_c = [], delta_c = [], obs_sig_c = [],
                   true_x1 = [], delta_x1 = [], obs_sig_x1 = [],
                   delta_mag = [], obs_sig_mag = [],
                   delta_mu = [],
                   obs_sig_mu = [],
                   dmudg = [],
                   dmudr = [],
                   dmudi = [],
                   dmudz = [],
                   LH = [],
                   outlier = [],
                   redshift = [],
                   resfl = [])


    allderivs = []
    for v1fl in glob.glob("*[0-9]_v1.txt"):
        [SNe] = readcol(v1fl, 'a')
        derivs = [item.replace("$UNION/", "") + "/result_deriv.dat" for item in SNe]
        allderivs += derivs

        
    for resfl in tqdm.tqdm(allderivs):
        resfl = resfl.replace("result_deriv.dat", "result_salt2.dat")

        obs_c = read_param(resfl, "Color")
        if obs_c != None:
            obs_sig_c = read_param(resfl, "Color", ind = 2)


            obs_x0 = read_param(resfl, "X0")
            obs_mag = -2.5*np.log10(obs_x0)
            obs_x1 = read_param(resfl, "X1")
            obs_sig_x1 = read_param(resfl, "X1", ind = 2)
            
            obs_sig_mag = (2.5/np.log(10.))*read_param(resfl, "X0", ind = 2)/obs_x0


            redshift = read_param(resfl, "Redshift")

            paramsfl = resfl + ""
            for red_key in "SLHV":
                for red_key2 in "OI":
                    paramsfl = paramsfl.replace("/SN" + red_key + red_key2, "/SN_params/params_").replace("/result_salt2.dat", ".dat").replace("dataset_", "UNITY_")
            

            print("resfl", resfl, "paramsfl", paramsfl)
            true_x0 = read_param(paramsfl, "x0")
            true_mag = -2.5*np.log10(true_x0)

            true_x1 = read_param(paramsfl, "x1")
            true_c = read_param(paramsfl, "c")

            all_dat["outlier"].append(read_param(paramsfl, "outlier"))

            all_dat["resfl"].append(resfl)

            all_dat["true_c"].append(true_c)
            all_dat["delta_c"].append(obs_c - true_c)
            all_dat["obs_sig_c"].append(obs_sig_c)

            if resfl.count("_L_"):
                LH = "L"
            elif resfl.count("_H_"):
                LH = "H"
            elif resfl.count("_V_"):
                LH = "V"
            elif resfl.count("_S_"):
                LH = "S"
            else:
                assert 0, resfl

            all_dat["LH"].append(LH)

            all_dat["delta_mag"].append(obs_mag - true_mag)
            all_dat["obs_sig_mag"].append(obs_sig_mag)

            all_dat["true_x1"].append(true_x1)
            all_dat["delta_x1"].append(obs_x1 - true_x1)
            all_dat["obs_sig_x1"].append(obs_sig_x1)


            all_dat["delta_mu"].append((obs_mag + global_alpha*obs_x1 - global_beta*obs_c) - (true_mag + global_alpha*true_x1 - global_beta*true_c))
            all_dat["obs_sig_mu"].append(read_param(resfl, "dmu_estimate"))


            all_dat["redshift"].append(redshift)

            dmudzps = get_dmu("griz", resfl)
            for band in "griz":
                all_dat["dmud" + band].append(dmudzps[band])

    for key in all_dat:
        all_dat[key] = np.array(all_dat[key])

    all_dat["pulls_mu"] = all_dat["delta_mu"]/all_dat["obs_sig_mu"]
    all_dat["pulls_mag"] = all_dat["delta_mag"]/all_dat["obs_sig_mag"]
    all_dat["pulls_c"] = all_dat["delta_c"]/all_dat["obs_sig_c"]
    all_dat["pulls_x1"] = all_dat["delta_x1"]/all_dat["obs_sig_x1"]


    pickle.dump(all_dat, open("all_dat.pickle", 'wb'))
    return all_dat


global_alpha = 0.15
global_beta = 3.1

print("[read or load] all_dat.pickle")

if sys.argv[1] == "read":
    all_dat = read_dat()
elif sys.argv[1] == "load":
    all_dat = pickle.load(open(sys.argv[2], 'rb'))
else:
    raise "Unknown option " + sys.argv[1]


print(len(all_dat["redshift"]))

for key in all_dat:
    print(key, all_dat[key])



plt.figure(figsize = (36, 32))

plt_ind = 1

for key in all_dat:
    if all_dat[key].dtype == np.float64:
        plt.subplot(5,5,plt_ind)
        inds = np.where((all_dat["LH"] == "L")*(all_dat["outlier"] == 0))
        counts, bins, NA = plt.hist(all_dat[key][inds], bins = 80, color = 'b')

        inds = np.where((all_dat["dmudg"] > -0.9)*(all_dat["LH"] == "L")*(all_dat["outlier"] == 0))
        the_med = np.median(all_dat[key][inds])
        the_unc = np.std(all_dat[key][inds], ddof=1)*np.sqrt(0.5*np.pi/len(all_dat[key][inds]))
        
        plt.hist(all_dat[key][inds], bins = bins, color = 'g', label = "Median %.2g +- %.2g" % (the_med, the_unc))

        inds = np.where((all_dat["dmudg"] > -0.85)*(all_dat["LH"] == "L")*(all_dat["outlier"] == 0))
        the_med = np.median(all_dat[key][inds])
        the_unc = np.std(all_dat[key][inds], ddof=1)*np.sqrt(0.5*np.pi/len(all_dat[key][inds]))

        plt.hist(all_dat[key][inds], bins = bins, color = 'orange', label = "Median %.2g +- %.2g" % (the_med, the_unc))

        inds = np.where((all_dat["dmudg"] > -0.8)*(all_dat["LH"] == "L")*(all_dat["outlier"] == 0))

        the_med = np.median(all_dat[key][inds])
        the_unc = np.std(all_dat[key][inds], ddof=1)*np.sqrt(0.5*np.pi/len(all_dat[key][inds]))
        plt.hist(all_dat[key][inds], bins = bins, color = 'r', label = "Median %.2g +- %.2g" % (the_med, the_unc))

        plt.legend(loc = 'best')

        plt.title(key)
        plt_ind += 1
        plt.yscale('log')
        
        
plt.tight_layout()
plt.savefig("high_dmudg.pdf", bbox_inches = 'tight')
plt.close()


plt.plot(all_dat["redshift"], all_dat["obs_sig_mu"], '.', alpha = 0.1)
plt.figtext(0.98, 0.991, "Simulated Data", color = 'r', ha = 'right', va = 'top', bbox=dict(edgecolor = 'r', pad = 1, facecolor = 'w'))
plt.savefig("sig_mu_vs_z.pdf")
plt.close()



all_dat["pulls_mu"] = all_dat["delta_mu"]/all_dat["obs_sig_mu"]
all_dat["weight_mu_with0.12"] = 1/(0.12**2. + all_dat["obs_sig_mu"]**2.)


for z_not_x1 in [0, 1]:
    for include_outlier in [0, 1]:
        plt.figure(1, figsize = (12, 5))
        nplt = 4

        for pltind, key in enumerate(["mu", "mag", "x1", "c"]):

            for LH in "LSHV":
                pltcolor = dict(S = 'm', L = 'b', H = 'g', V = 'r')[LH]
                pltsymb = dict(S = 'v', L = '.', H = '^', V = '*')[LH]
                pltlabel = dict(S = "Low-$z$ $UBV$", L = "Low-$z$ $ugriz$", H = "Mid-$z$", V = "High-$z$")[LH]

                if include_outlier:
                    outlier_mask = np.ones(len(all_dat["outlier"]))
                else:
                    outlier_mask = all_dat["outlier"] == 0

                inds = np.where((all_dat["LH"] == LH)*outlier_mask)
                if z_not_x1:
                    zs = all_dat["redshift"][inds]
                else:
                    zs = all_dat["true_x1"][inds]

                    
                bin_edges = scoreatpercentile(zs, np.linspace(0, 100, int(len(zs)/400.)))
                bin_edges[0] -= 0.001
                bin_edges[-1] += 0.001

                print("bin_edges", bin_edges)

                for i in range(len(bin_edges) - 1):
                    if z_not_x1:
                        inds = np.where((all_dat["LH"] == LH)*(all_dat["redshift"] >= bin_edges[i])*(all_dat["redshift"] < bin_edges[i+1])*outlier_mask)
                    else:
                        inds = np.where((all_dat["LH"] == LH)*(all_dat["true_x1"] >= bin_edges[i])*(all_dat["true_x1"] < bin_edges[i+1])*outlier_mask)

                        
                    rms = np.std(all_dat["pulls_" + key][inds], ddof=1)
                    uncrms = rms/np.sqrt(2.*len(inds[0]))

                    mean = np.mean(all_dat["pulls_" + key][inds])
                    uncmean = rms/np.sqrt(len(inds[0]))

                    mean_bin = 0.5*(bin_edges[i] + bin_edges[i+1])
                    #plt.plot(mean_bin, rms, '.', color = pltcolor)
                    #plt.plot([mean_bin]*2, [rms - uncrms - 0.9, rms + uncrms - 0.9], color = pltcolor)
                    plt.figure(1)
                    plt.subplot(2, nplt, 1 + pltind)
                    plt.plot(mean_bin, mean, pltsymb, color = pltcolor, label = (i == 0)*pltlabel)
                    plt.subplot(2, nplt, nplt + 1 + pltind)
                    plt.plot(mean_bin, rms, pltsymb, color = pltcolor)

            plt.figure(1)
            plt.subplot(2, nplt, 1 + pltind)
            if pltind == 0:
                plt.legend(loc = 'upper right', bbox_to_anchor = (1.05, 1.05))

            plt.axhline(0, color = 'k', linewidth = 0.8)
            plt.title(dict(mu = "$m_B + %.2f %s x_1 - %.2f %s c$" % (global_alpha, '\,', global_beta, '\,'),
                           mag = "$m_B$", x1 = "$x_1$", c = "$c$")[key])

            if z_not_x1:
                plt.xscale('log')
            plt.subplot(2, nplt, nplt + 1 + pltind)
            plt.axhline(1, color = 'k', linewidth = 0.8)
            if z_not_x1:
                plt.xscale('log')
            plt.xlabel("Sim LC Redshift Bin")
            if pltind == 0:
                plt.subplot(2, nplt, 1 + pltind)
                plt.ylabel("Sim LC Mean Pull,\nEqual Number per Bin")
                plt.subplot(2, nplt, nplt + 1 + pltind)
                plt.ylabel("Sim LC RMS Pull,\nEqual Number per Bin")


        fig = plt.figure(1)
        fig.align_ylabels()

        plt.tight_layout()
        plt.figtext(0.98, 1.00, "Simulated Data", color = 'r', ha = 'right', va = 'top', bbox=dict(edgecolor = 'r', pad = 1, facecolor = 'w'))

        plt.savefig("LC_compare_pulls_incloutl=%i_vs_%s.pdf" % (include_outlier, ["x1", "z"][z_not_x1]), bbox_inches = 'tight')
        plt.close()



plt.figure()

inds = np.where(all_dat["outlier"] == 1)

plt.subplot(2,1,1)
plt.plot(all_dat["redshift"][inds], all_dat["true_c"][inds], '.', color = 'b')
plt.subplot(2,1,2)
plt.plot(all_dat["redshift"][inds], all_dat["true_x1"][inds], '.', color = 'b')
plt.figtext(0.98, 0.991, "Simulated Data", color = 'r', ha = 'right', va = 'top', bbox=dict(edgecolor = 'r', pad = 1, facecolor = 'w'))

plt.savefig("outlier_populations.pdf", bbox_inches = 'tight')
plt.close()


plt.figure(figsize = (48, 32))

zbins = 400

for i, keys in enumerate([("redshift", "delta_mag", 0, 0),
                          ("redshift", "delta_c", 0, 0),
                          ("true_c", "delta_c", 0, 0),
                          ("redshift", "delta_x1", 0, 0),
                          ("redshift", "pulls_c", 0, 0),
                          ("redshift", "pulls_x1", 0, 0),
                          ("redshift", "delta_mu", 0, 0),
                          ("redshift", "delta_mag", zbins, 1),
                          ("redshift", "delta_c", zbins, 1),
                          ("redshift", "true_c", zbins, 1),
                          ("redshift", "obs_sig_mu", zbins, 1),
                          ("redshift", "delta_mu", 35, 1),
                          ("true_c", "delta_mu", 35, 1),
                          ("true_x1", "delta_mu", 35, 1),
                          ("true_c", "delta_c", 35, 1),
                          ("redshift", "delta_x1", zbins, 1),
                          ("redshift", "delta_mu", zbins, 1),
                          ("redshift", "dmudg", zbins, 1),
                          ("redshift", "dmudg", zbins, 1),
                          ("dmudg", "delta_mag", 35, 1),
                          ("dmudg", "delta_x1", 35, 1),
                          ("dmudg", "delta_c", 35, 1),
                          ("dmudg", "delta_mu", 35, 1),
                          ("dmudr", "delta_mu", 35, 1),
                          ("dmudi", "delta_mu", 35, 1),
                          ("dmudz", "delta_mu", 35, 1),
                          ("obs_sig_mu", "delta_mu", 35, 1),
                          ("obs_sig_c", "delta_mu", 35, 1),
                          ("obs_sig_c", "delta_c", 35, 1),
                          ("true_x1", "delta_x1", 35, 1)]):
    
    plt.subplot(6,5,i+1)
    if keys[2] == 0:
        plt.plot(all_dat[keys[0]], all_dat[keys[1]], '.', label = "Mean %.3f +- %.3f Median %.3f RMS %.3f" % (np.mean(all_dat[keys[1]]), np.std(all_dat[keys[1]], ddof=1)/np.sqrt(float(len(all_dat[keys[1]]))),
                                                                                                              np.median(all_dat[keys[1]]),
                                                                                                              np.std(all_dat[keys[1]], ddof=1)), color = 'b', alpha = 0.05)#, gridsize=100)
        
    else:
        bin_by_sample = keys[3]
        
        for LH in "LSHV"*bin_by_sample + "A"*(1 - bin_by_sample):
            pltcolor = dict(L = 'b', S = 'm', H = 'g', V = 'r', A = 'k')[LH]

            if bin_by_sample:
                inds = np.where((all_dat["redshift"] > -1)*(all_dat["LH"] == LH)*(all_dat["outlier"] == 0))#*(all_dat["redshift"] < 0.055))
            else:
                inds = np.where((all_dat["redshift"] > -1)*(all_dat["outlier"] == 0))#*(all_dat["redshift"] < 0.055))
                
            nsne = len(all_dat[keys[0]][inds])
            binx, biny = do_med_bins(all_dat[keys[0]][inds], all_dat[keys[1]][inds], np.ones(nsne, dtype=np.float64), keys[2])
            plt.plot(binx, biny, '.', color = pltcolor, label = "Median")
           
            binx, biny = do_med_bins(all_dat[keys[0]][inds], all_dat[keys[1]][inds], np.ones(nsne, dtype=np.float64), keys[2], average_not_median = 1)
            plt.plot(binx, biny, '^', color = pltcolor, label = "Average")
            
            try:
                all_dat["obs_sig_" + keys[1].split("_")[-1]]
                has_uncs = 1
            except:
                has_uncs = 0
                
            if has_uncs:
                binx, biny = do_med_bins(all_dat[keys[0]][inds], all_dat[keys[1]][inds], all_dat["obs_sig_" + keys[1].split("_")[-1]][inds], keys[2], average_not_median = 1)
                plt.plot(binx, biny, '*', color = pltcolor, label = "Weighted")
                
                xlim = plt.xlim()
                pltx = np.linspace(0.01, xlim[1], 200)
                
                if keys[0] == "redshift" and keys[1] == "delta_mu":
                    #plty, label = fit_delta_cosmo(zs = binx, delta_mus = biny, pltzs = pltx, mu_uncs = np.ones(len(binx), dtype=np.float64), fit_Om = 1, fit_w0 = 0, fit_wa = 0)
                    #plt.plot(pltx, plty, label = label)
                    #plty, label = fit_delta_cosmo(zs = binx, delta_mus = biny, pltzs = pltx, mu_uncs = np.ones(len(binx), dtype=np.float64), fit_Om = 1, fit_w0 = 1, fit_wa = 0)
                    #plt.plot(pltx, plty, label = label)
                    pass
                

        plt.legend(loc = 'best')

        
    if (keys[0] == "redshift") or (keys[0] == "obs_sig_mu") or (keys[0] == "obs_sig_c"):
        plt.xscale('log')
        #plt.xlim(0.01, 3)

    plt.axhline(0)
    
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])

            
plt.tight_layout()
plt.figtext(0.98, 0.991, "Simulated Data", color = 'r', ha = 'right', va = 'top', bbox=dict(edgecolor = 'r', pad = 1, facecolor = 'w'))
plt.savefig("compare_LC_vs_input.pdf", bbox_inches = 'tight')
plt.close()



plt.figure(2, figsize = (5, 7))
for pltind, LH in enumerate(["SLHV", "H"]):
    LH_mask = np.array([LH.count(item) for item in all_dat["LH"]])
    inds = np.where(LH_mask*(all_dat["redshift"] > 0.0)*(all_dat["outlier"] == 0))
    zs = all_dat["redshift"][inds]
    delta_mu = all_dat["delta_mu"][inds]

    all_weights = all_dat["obs_sig_mu"][inds]**-2.    
    pltzs = np.linspace(0.01, 3, 300)

    pltys, thelabel = fit_delta_cosmo(zs = zs, delta_mus = all_dat["delta_mu"][inds], mu_uncs = 1./np.sqrt(all_dat["weight_mu_with0.12"][inds]), pltzs = pltzs,
                                      fit_Om = (LH == "H") | (LH == "SLHV"), fit_w0 = (LH == "LHV"), fit_wa = (LH == "LHV"), verbose = True)

    """
    pltys_unweight, thelabel_unweight = fit_delta_cosmo(zs = zs, delta_mus = all_dat["delta_mu"][inds], mu_uncs = np.ones(len(inds[0]), dtype=np.float64), pltzs = pltzs,
                                                        fit_Om = (LH == "H"), fit_w0 = (LH == "LHV"), fit_wa = (LH == "LHV"), verbose = True)
    """
    
    z_sort_inds = np.argsort(zs)


    if LH == "H":
        n_bins = 250
    else:
        n_bins = 1000


    target_weight_per_bin = sum(all_weights)/n_bins
    
    for i in tqdm.trange(n_bins + 1):
        tot_z = 0.
        tot_delta_mu = 0.
        tot_weight = 0.
        n_count = 0
        
        while tot_weight < target_weight_per_bin and len(z_sort_inds) > 0:
            tot_weight += all_weights[z_sort_inds[-1]]
            n_count += 1
            tot_z += zs[z_sort_inds[-1]]
            tot_delta_mu += delta_mu[z_sort_inds[-1]]

            z_sort_inds = z_sort_inds[:-1]
                        
        plt.figure(2)
        plt.subplot(2, 1, 1 + pltind)
        if n_count > 0:
            plt.plot(tot_z/n_count, tot_delta_mu/n_count, '.', color = 'k')
        print("Remaining", len(z_sort_inds))

    plt.figure(2)
    plt.subplot(2, 1, 1 + pltind)
    plt.axhline(0, color = 'k', linewidth = 0.8)
    plt.xlabel("Sim LC Redshift Bin")
    plt.ylabel("Binned $\Delta (m_B + %.2f %s x_1 - %.2f %s c)$\nfrom Sim LCs,\nEqual Weight per Bin" % (global_alpha, '\,', global_beta, '\,'))
    if pltind == 0:
        plt.xscale("log")

    xlim = list(plt.xlim())
    xlim[0] = max(xlim[0], 0)

    plt.plot(pltzs, pltys, label = thelabel, color = 'b')
    #plt.plot(pltzs, pltys_unweight, label = thelabel_unweight)
    plt.legend(loc = 'best')
    plt.xlim(xlim)
    


plt.figure(2)
plt.figtext(0.98, 0.92, "Simulated Data", color = 'r', ha = 'right', va = 'top', bbox=dict(edgecolor = 'r', pad = 1, facecolor = 'w'))
plt.savefig("sim_mean_resid.pdf", bbox_inches = 'tight')
plt.close()
