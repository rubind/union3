import numpy as np
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
import sys
import pickle
from matplotlib import rcParams
from cosmo_functions import no_big_bang, get_colors, get_DETF
from DavidsNM import miniNM_new
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True




def find_three_sigma(all_grids, key, top_not_bottom, chi2_val = 11.8292):
    no_nan = all_grids[key][2]
    no_nan[np.isnan(all_grids[key][2])] = 1000
    min_chi2 = np.min(no_nan, axis = 1) # Shape is y, x
    
    if top_not_bottom:
        ind_vertical = np.where(min_chi2 <= chi2_val)[0][-1]
    else:
        ind_vertical = np.where(min_chi2 <= chi2_val)[0][0]

    ind_horizontal = np.argmin(all_grids[key][2][ind_vertical])

    return all_grids[key][0][ind_horizontal], all_grids[key][1][ind_vertical]

def label_chi2(P, passdata):
    label_dict = passdata[0]

    chi2 = sum((P - label_dict["ys"])**2.)
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            diff = (P[i] - P[j])/label_dict["minys"]
            
            chi2 += np.exp(-diff**4)

    return chi2

def rgb_to_gray(rgb):
    return 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]


def optimize_label_positions(label_dict):
    P, NA, NA = miniNM_new(ministart = label_dict["ys"], miniscale = [label_dict["minys"]]*len(label_dict["ys"]), chi2fn = label_chi2, passdata = label_dict, verbose = False, compute_Cmat = False)

    for i in range(len(P)):
        if len(label_dict["labels"][i]) > 0:
            plt.text(label_dict["xs"][i], P[i], label_dict["labels"][i], color = label_dict["colors"][i], fontsize = label_dict["sizes"][i], ha = label_dict["has"][i], va = label_dict["vas"][i], zorder = 10)


def make_contours(all_grids, BAO_Omh2 = 0, show_all_four = 0, show_H0s = 0, sigma_level = None, combined_and_SNe = 0):
    DETF_FoM_txt = ""
    
    if all_grids["model"] == "flatwCDM" or (all_grids["model"] == "flatLCDM"):
        plt.figure(figsize = (5,5))
    elif (all_grids["model"].replace("EDE", "") == "flatw0wa") or (all_grids["model"].replace("EDE", "") == "w0wa"):
        plt.figure(figsize = (5,5))
        DETF_FoM = get_DETF(all_grids["SNeBAOCMB"])
        DETF_FoM_txt = "\nDETF FoM SNe BAO CMB: %.2f" % DETF_FoM

        
    elif all_grids["model"] == "LCDM":
        plt.figure(figsize = (5,7.5))
        plt.axes().set_aspect("equal")

    else:
        assert 0


    if "BAO" in all_grids:
        BAO_key = "BAO" + "_Omh2"*BAO_Omh2
        plt.contourf(all_grids[BAO_key][0], all_grids[BAO_key][1], all_grids[BAO_key][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("green"), zorder = 0 + 2.5*(all_grids["model"] == "LCDM"))
        plt.contourf(all_grids["CMB"][0], all_grids["CMB"][1], all_grids["CMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("orange"), zorder = 1)
        plt.contourf(all_grids["SNe"][0], all_grids["SNe"][1], all_grids["SNe"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("blue"), zorder = 2)
        plt.contourf(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("gray"), zorder = 3)
        
        
        plt.contour(all_grids["SNe"][0], all_grids["SNe"][1], all_grids["SNe"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25, zorder = 4)
        plt.contour(all_grids["CMB"][0], all_grids["CMB"][1], all_grids["CMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25, linestyles = "dotted", zorder = 5)
        plt.contour(all_grids[BAO_key][0], all_grids[BAO_key][1], all_grids[BAO_key][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25, linestyles = "dashed", zorder = 6)
    else:
        if BAO_Omh2:
            return 0
        
        if show_all_four == 0 and show_H0s == 0 and combined_and_SNe == 0:
            plt.contourf(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("teal"))
            plt.contour(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25)
            label_dict = dict(xs = [-1.2], ys = [-0.25], labels = ["$\Lambda$CDM"], colors = ["k"], sizes = [12]*10, has = ["center"]*10, vas = ["center"]*10, minys = 0.14)
            optimize_label_positions(label_dict)
        elif show_all_four > 0:
            level_for_four = [None, 2.29575, 6.18007][show_all_four]
            #plt.contour(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, level_for_four], colors = 'k', linewidths = 0.25)

            label_dict = dict(xs = [-1.2], ys = [-0.25], labels = ["$\Lambda$CDM"], colors = ["k"], sizes = [12]*10, has = ["center"]*10, vas = ["center"]*10, minys = 0.14)


            for gridkey in ["SNeBAO", "SNeCMB", "BAOCMB", "SNeBAOCMB"]:
                plt.contour(all_grids[gridkey][0], all_grids[gridkey][1], all_grids[gridkey][2], levels = [0, level_for_four], colors = 'k', linewidths = 0.25, zorder = 5)

                if gridkey == "SNeBAOCMB":
                    these_colors = get_colors("teal")[0]
                else:
                    these_colors = get_colors(gridkey.replace("SNe", "blue").replace("CMB", "orange").replace("BAO", "green"))[0]

                plt.contourf(all_grids[gridkey][0], all_grids[gridkey][1], all_grids[gridkey][2], levels = [0, level_for_four], colors = [these_colors])

                
                if len(gridkey) < 9:
                    three_x, three_y = find_three_sigma(all_grids, key = gridkey, top_not_bottom = 1, chi2_val = level_for_four)
                    gridkeylabel = gridkey.replace("SNe", "SNe+").replace("BAO", "BAO+").replace("CMB", "CMB+")
                    gridkeylabel = gridkeylabel[:-1]

                    if gridkeylabel == "SNe+BAO":
                        gridkeylabel = "SNe+BAO+$\omega_b$"
                    
                    label_dict["xs"].append(three_x - 0.2)
                    label_dict["ys"].append(three_y + 0.25)
                    label_dict["labels"].append(gridkeylabel)
                    label_dict["colors"].append(these_colors)

                    
            #plt.contourf(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292][:3], colors = get_colors("teal")[:3])
            #plt.contour(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292][:3], colors = 'k', linewidths = 0.25)

            optimize_label_positions(label_dict)
            
            plt.text(-1.75, -2.5, [None, "1", "2"][show_all_four] + "$\sigma$ Contours ($\chi^2 - \chi^2_{\mathrm{min}} < %.2f)$" % level_for_four, va = 'center', ha = 'left')
            
        elif show_H0s > 0:
            level_for_four = [None, 2.29575, 6.18007][show_H0s]
            #plt.contour(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, level_for_four], colors = 'k', linewidths = 0.25)

            label_dict = dict(xs = [-1.2], ys = [-0.25], labels = ["$\Lambda$CDM"], colors = ["k"], sizes = [12]*10, has = ["center"]*10, vas = ["center"]*10, minys = 0.14)

            
            for gridkey in ["SNeBAOCMB", "SNeBAOCMBH0T", "SNeBAOCMBH0C"]:
                plt.contour(all_grids[gridkey][0], all_grids[gridkey][1], all_grids[gridkey][2], levels = [0, level_for_four], colors = 'k', linewidths = 0.25, zorder = 5)

                if gridkey == "SNeBAOCMB":
                    these_colors = get_colors("teal")[0]
                elif gridkey == "SNeBAOCMBH0T":
                    these_colors = get_colors("blueorange")[0]
                elif gridkey == "SNeBAOCMBH0C":
                    these_colors = get_colors("greenorange")[0]
                else:
                    print("Unknown", gridkey)

                plt.contourf(all_grids[gridkey][0], all_grids[gridkey][1], all_grids[gridkey][2], levels = [0, level_for_four], colors = [these_colors])

                
                three_x, three_y = find_three_sigma(all_grids, key = gridkey, top_not_bottom = 1, chi2_val = level_for_four)
                gridkeylabel = gridkey.replace("SNe", "SNe+").replace("BAO", "BAO+").replace("CMB", "CMB+").replace("H0T", "$H_0^{\mathrm{TRGB}}$+").replace("H0C", "$H_0^{\mathrm{Ceph.}}$+")
                gridkeylabel = gridkeylabel[:-1]
                
                label_dict["xs"].append(three_x - 0.1)
                label_dict["ys"].append(three_y + 0.25)
                label_dict["labels"].append(gridkeylabel)
                label_dict["colors"].append(these_colors)

            #plt.contourf(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292][:3], colors = get_colors("teal")[:3])
            #plt.contour(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292][:3], colors = 'k', linewidths = 0.25)

            optimize_label_positions(label_dict)
            
            plt.text(-1.75, -2.5, [None, "1", "2"][show_H0s] + "$\sigma$ Contours ($\chi^2 - \chi^2_{\mathrm{min}} < %.2f)$" % level_for_four, va = 'center', ha = 'left')
        elif combined_and_SNe:
            for gridkey in ["SNe", "SNeBAOCMB"]:
                plt.contour(all_grids[gridkey][0], all_grids[gridkey][1], all_grids[gridkey][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25, zorder = 5)
                if gridkey == "SNeBAOCMB":
                    these_colors = get_colors("teal")
                else:
                    these_colors = get_colors(gridkey.replace("SNe", "blue").replace("CMB", "orange").replace("BAO", "green"))
                    
                plt.contourf(all_grids[gridkey][0], all_grids[gridkey][1], all_grids[gridkey][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = these_colors)


    if all_grids["model"] == "flatwCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$w$")
        plt_name = "Om-w%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)

        three_x, three_y = find_three_sigma(all_grids, key = "SNe", top_not_bottom = 0)
        plt.text(three_x, three_y - 0.07, "SNe", color = get_colors("blue")[0], ha = 'center', va = 'center')

        three_x, three_y = find_three_sigma(all_grids, key = BAO_key, top_not_bottom = 0)
        plt.text(three_x, three_y - 0.07, BAO_key, color = get_colors("green")[0], ha = 'center', va = 'center')

        plt.text(0.20, -1.15, "CMB", color = get_colors("orange")[0])
        
    elif all_grids["model"] == "flatLCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$h$")
        plt_name = "Om-h%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)
        
    elif all_grids["model"].replace("EDE", "") == "flatw0wa" or all_grids["model"].replace("EDE", "") == "w0wa":
        plt.xlabel("$w_0$")
        plt.ylabel("$w_a$")

        if all_grids["model"].replace("EDE", "") == "flatw0wa":
            plt_name = "w0-wa%s%s%s%s.pdf" % ("_BAO_Omh2"*BAO_Omh2, "_all4"*show_all_four, "_H0s"*show_H0s, "_SNeComb"*combined_and_SNe)
        else:
            plt_name = "w0-wa_open%s%s%s%s.pdf" % ("_BAO_Omh2"*BAO_Omh2, "_all4"*show_all_four, "_H0s"*show_H0s, "_SNeComb"*combined_and_SNe)
            
        xlim = plt.xlim()
        ylim = plt.ylim()
        early_x = np.linspace(xlim[0], xlim[-1], 20)

        # wa -> -0.302669 - 1.16638 w0 % DE is 1% of matter density at z=1100
        # -0.174851 - 1.16638 w0 % DE is 10% of matter density at z=1100

        # -0.0470338 - 0.127817 n - 1.16638 w0 for 10^(-n)
        
        plt.fill_between(early_x, -0.302669 - 1.16638*early_x, [3]*len(early_x), color = (0.8, 0.8, 0.8), zorder = 0)
        plt.fill_between(early_x, -0.174851 - 1.16638*early_x, [3]*len(early_x), color = (0.7, 0.7, 0.7), zorder = 0.1)
        plt.xlim([-2, 0])
        plt.ylim([-3, 2])
        plt.text(-0.75, 1.7, "Early Matter Domination Violated", color = 'k', ha = 'center', va = 'center')#, bbox=dict(facecolor='w', edgecolor = 'w'))

        plt.plot(-1, 0., '.', color = 'k')

        
    elif all_grids["model"] == "LCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$\Omega_{\Lambda}$")

        no_big_bang_x = np.linspace(0., 1., 200)
        plt.fill_between(no_big_bang_x, no_big_bang(no_big_bang_x), [2]*len(no_big_bang_x), color = (0.7, 0.7, 0.7))
        plt.plot([0, 1], [1, 0], color = 'k', linewidth = 0.75)
        plt.text(0.7, 0.3, "Flat Universe", color = 'k', rotation = -45, ha = 'center', va = 'center', bbox=dict(facecolor='w', edgecolor = 'w'))
        plt.text(0.07, 1.42, "No\nBig\nBang", color = 'k', rotation = 65, ha = 'center', va = 'center')#, bbox=dict(facecolor='w', edgecolor = 'w'))
        plt.text(0.65, 0.45, "CMB", color = get_colors("orange")[0])

        three_x, three_y = find_three_sigma(all_grids, key = "SNe", top_not_bottom = 1)
        plt.text(three_x, three_y + 0.03, "SNe", color = get_colors("blue")[0], ha = 'center', va = 'center')

        three_x, three_y = find_three_sigma(all_grids, key = BAO_key, top_not_bottom = 1)
        plt.text(three_x, three_y + 0.03, BAO_key, color = get_colors("green")[0], ha = 'center', va = 'center')

        
        plt.xlim(0, 1)
        plt.ylim(0, 1.5)


        
        plt_name = "Om-OL%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)

    else:
        assert 0

    all_txt = "All: " + str(all_grids["SNeBAOCMB_fit"]) + " " + str(np.sqrt(np.diag(all_grids["SNeBAOCMB_cmat"]))) + '\n'
    all_txt += "SN+CMB: " + str(all_grids["SNeCMB_fit"]) + " " + str(np.sqrt(np.diag(all_grids["SNeCMB_cmat"]))) + '\n'
    all_txt += "BAO+CMB: " + str(all_grids["BAOCMB_fit"]) + " " + str(np.sqrt(np.diag(all_grids["BAOCMB_cmat"])))
    all_txt += DETF_FoM_txt

    plt_name = plt_name.replace(".pdf", "_" + fl.split(".")[0] + ".pdf")
    plt.savefig(plt_name, bbox_inches = 'tight', metadata=dict(Keywords = all_txt))
    plt.close()
    
    
def make_latex_table(all_grids):
    keys_to_look_for = ["SNe_minos", "SNeCMB_minos", "BAOCMB_minos", "SNeBAO_minos", "SNeBAOCMB_minos", "SNeBAOCMBH0T_minos", "SNeBAOCMBH0C_minos"]
    labels = dict(SNe_minos = "SNe", SNeBAO_minos = "SNe+BAO+$\omega_b$", SNeCMB_minos = "SNe+CMB", BAOCMB_minos = "BAO+CMB", SNeBAOCMB_minos = "SNe+BAO+CMB", SNeBAOCMBH0T_minos = "SNe+BAO+CMB+$H_0^{\mathrm{TRGB}}$",
                  SNeBAOCMBH0C_minos = "SNe+BAO+CMB+$H_0^{\mathrm{Ceph.}}$")
    param_order = [["h"], ["Om"], ["Ok"], ["w", "w0"], ["wa"]]
    fmt_strs = ["%.3f", "%.3f", "%.3f", "%.3f", "%.2f"]

    these_latex_lines = ""
    
    for key in keys_to_look_for:
        if key in all_grids:
            add_phantom = all_grids[key.replace("_minos", "_chi2")] < 9.95
            add_phantom = "\phantom{0}"*add_phantom

            these_latex_lines += labels[key] + " & %s%.1f (%i) " % (add_phantom, all_grids[key.replace("_minos", "_chi2")],
                                                                    all_grids[key.replace("_minos", "_n_data")] - all_grids[key.replace("_minos", "_n_par")])

            for possible_params, fmt_str in zip(param_order, fmt_strs):
                found_one = 0
                
                for param in possible_params:
                    huge_Om_uncs = all_grids[key]["Om"][1] - all_grids[key]["Om"][2] > 0.25
                    if param in all_grids[key]:
                        #print("this_conf", all_grids[key][param], key, param)

                        this_conf = all_grids[key][param]
                        if (huge_Om_uncs == 0) or (["Ok", "Om", "h"].count(param) == 0):
                            add_phantom = (param == "Ok")*(this_conf[0] >= 0)
                            add_phantom = "\phantom{-}"*add_phantom
                            
                            these_latex_lines += (" &  $" + add_phantom + fmt_str + "^{+" + fmt_str + "}_{" + fmt_str + "}$ ") % tuple(this_conf)
                        else:
                            these_latex_lines += (" &  $" + fmt_str + "$ ") % this_conf[0]
                        found_one += 1
                if found_one == 0:
                    these_latex_lines += " & \\nodata "
                elif found_one > 1:
                    assert 0, "Conflicting params found!"


            
            if all_grids["model"].count("w0wa"):
                gridkey = key.replace("_minos", "")
                try:
                    all_grids[gridkey][2][0,0]
                    do_DETF = 1
                except:
                    do_DETF = 0

                if do_DETF:
                    print("DETF", gridkey, key)
                    DETF_FoM = get_DETF(all_grids[gridkey])
                    if DETF_FoM > 0:
                        these_latex_lines += " & %.2f " % DETF_FoM
                    else:
                        these_latex_lines += " & \\nodata "                    
                else:
                    these_latex_lines += " & \\nodata "
            else:
                these_latex_lines += " & \\nodata "
            these_latex_lines += ' \\\\ \n'
    return these_latex_lines



model_labels = dict(flatLCDM = "Flat $\Lambda$CDM",
                    w0wa = "Open $w_0$-$w_a$, No EDE",
                    flatw0wa = "Flat $w_0$-$w_a$, No EDE",
                    w0waEDE = "Open $w_0$-$w_a$",
                    flatw0waEDE = "Flat $w_0$-$w_a$",
                    LCDM = "Open $\Lambda$CDM",
                    flatwCDM = "Flat $w$CDM")

all_latex_lines = ""

for fl in sys.argv[1:]:
    all_grids = pickle.load(open(fl, 'rb'))

    print(fl, all_grids.keys())
    
    all_latex_lines += "\hline\n \multicolumn{8}{c}{" + model_labels[all_grids["model"]] + "}\\\\ \n \hline\n"
    all_latex_lines += make_latex_table(all_grids)
    
    #make_contours(all_grids, BAO_Omh2 = 0)
    #make_contours(all_grids, BAO_Omh2 = 1)
    #make_contours(all_grids, show_all_four = 1)
    #make_contours(all_grids, show_all_four = 2)
    #make_contours(all_grids, show_H0s = 1)
    #make_contours(all_grids, show_H0s = 2)
    #make_contours(all_grids, combined_and_SNe = 1)

print(all_latex_lines)
