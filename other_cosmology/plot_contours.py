import numpy as np
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
import sys
import pickle
from matplotlib import rcParams
from cosmo_functions import no_big_bang, get_colors
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True



def get_DETF(the_grid):
    dx = the_grid[0][1:] - the_grid[0][:-1]
    assert np.isclose(dx, dx[0]).all()
    dy = the_grid[1][1:] - the_grid[1][:-1]
    assert np.isclose(dy, dy[0]).all()


    plt.figure(2)

    if np.any(the_grid[2][0, :] < 7):
        return 0
    if np.any(the_grid[2][-1, :] < 7):
        return 0
    if np.any(the_grid[2][:, 0] < 7):
        return 0
    if np.any(the_grid[2][:, -1] < 7):
        return 0

    
    
    for cut_val in np.linspace(6.15, 6.21, 100):
        included_points = float((the_grid[2] <= cut_val).sum())
        included_area = included_points*dx[0]*dy[0]

        plt.plot(cut_val, 1./included_area, '.', color = 'b')
    plt.axvline(6.18007)
    plt.savefig("DETF_evaluation.pdf")
    plt.close()
        
    included_points = float((the_grid[2] <= 6.18007).sum())
    included_area = included_points*dx[0]*dy[0]
    
    DETF_FoM = 1./included_area
    return DETF_FoM

def find_three_sigma(all_grids, key, top_not_bottom):
    no_nan = all_grids[key][2]
    no_nan[np.isnan(all_grids[key][2])] = 1000
    min_chi2 = np.min(no_nan, axis = 1) # Shape is y, x
    
    if top_not_bottom:
        ind_vertical = np.where(min_chi2 <= 11.8292)[0][-1]
    else:
        ind_vertical = np.where(min_chi2 <= 11.8292)[0][0]

    ind_horizontal = np.argmin(all_grids[key][2][ind_vertical])

    return all_grids[key][0][ind_horizontal], all_grids[key][1][ind_vertical]

def make_contours(all_grids, BAO_Omh2):
    DETF_FoM_txt = ""
    
    if all_grids["model"] == "flatwCDM" or (all_grids["model"] == "flatLCDM"):
        plt.figure(figsize = (5,5))
    elif (all_grids["model"] == "flatw0wa") or (all_grids["model"] == "w0wa"):
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
        
        plt.contourf(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("teal"))
        plt.contour(all_grids["SNeBAOCMB"][0], all_grids["SNeBAOCMB"][1], all_grids["SNeBAOCMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25)

        

    if all_grids["model"] == "flatwCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$w$")
        plt_name = "Om-w%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)

    elif all_grids["model"] == "flatLCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$h$")
        plt_name = "Om-h%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)
        
    elif all_grids["model"] == "flatw0wa" or all_grids["model"] == "w0wa":
        plt.xlabel("$w_0$")
        plt.ylabel("$w_a$")

        if all_grids["model"] == "flatw0wa":
            plt_name = "w0-wa%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)
        else:
            plt_name = "w0-wa_open%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)
                    
        xlim = plt.xlim()
        ylim = plt.ylim()
        early_x = np.linspace(xlim[0], xlim[-1], 20)

        # wa -> -0.302669 - 1.16638 w0 % DE is 1% of matter density at z=1100
        # -0.174851 - 1.16638 w0 % DE is 10% of matter density at z=1100

        # -0.0470338 - 0.127817 n - 1.16638 w0 for 10^(-n)
        
        plt.fill_between(early_x, -0.302669 - 1.16638*early_x, [3]*len(early_x), color = (0.8, 0.8, 0.8))
        plt.fill_between(early_x, -0.174851 - 1.16638*early_x, [3]*len(early_x), color = (0.7, 0.7, 0.7))
        plt.xlim([-2, 0])
        plt.ylim([-3, 2])
        plt.text(-0.75, 1.5, "Early Matter Domination Violated", color = 'k', ha = 'center', va = 'center')#, bbox=dict(facecolor='w', edgecolor = 'w'))

        plt.plot(-1, 0., '.', color = 'k')
        plt.text(-0.95, 0.2, "$\Lambda$CDM", color = 'k', ha = 'left', va = 'center')#, bbox=dict(facecolor='w', edgecolor = 'w'))

        
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
            these_latex_lines += labels[key] + " & %.1f " % all_grids[key.replace("_minos", "_chi2")]

            for possible_params, fmt_str in zip(param_order, fmt_strs):
                found_one = 0
                
                for param in possible_params:
                    if param in all_grids[key]:
                        this_conf = all_grids[key][param]
                        these_latex_lines += (" &  $" + fmt_str + "^{+" + fmt_str + "}_{" + fmt_str + "}$ ") % tuple(this_conf)
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
                    w0wa = "Open $w_0$-$w_a$",
                    flatw0wa = "Flat $w_0$-$w_a$",
                    LCDM = "Open $\Lambda$CDM",
                    flatwCDM = "Flat $w$CDM")

all_latex_lines = ""

for fl in sys.argv[1:]:
    all_grids = pickle.load(open(fl, 'rb'))

    print(fl, all_grids.keys())
    
    all_latex_lines += "\hline\n \multicolumn{8}{c}{" + model_labels[all_grids["model"]] + "}\\\\ \n \hline\n"
    all_latex_lines += make_latex_table(all_grids)
    
    make_contours(all_grids, BAO_Omh2 = 0)
    make_contours(all_grids, BAO_Omh2 = 1)

print(all_latex_lines)
