import numpy as np
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
import sys
import pickle
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True


def get_colors(key):
    if key == "blue":
        return ((0., 83/255., 152/255.),
                (30/255., 104/255., 168/255.),
                (92/255., 140/255., 190/255.))

    if key == "orange":
        return ((243/255., 116/255., 17/255.),
                (248/255., 144/255., 62/255.),
                (250/255., 180/255., 110/255.))

    if key == "green":
        return ((0., 160/255., 52/255.),
                (50/255., 176/255., 86/255.),
                (117/255., 198/255., 126/255.))
              
    if key == "gray":
        return ((71/255., 71/255., 71/255.),
                (119/255., 119/255., 119/255.),
                (178/255., 178/255., 178/255.))

    if key == "teal":
        return ((0/255., 168/255., 168/255.),
                (127/255., 204/255., 189/255.),
                (190/255., 226/255., 210/255.))
    assert 0, key


def get_DETF(all_grids):
    dx = all_grids["Combined"][0][1:] - all_grids["Combined"][0][:-1]
    assert np.isclose(dx, dx[0]).all()
    dy = all_grids["Combined"][1][1:] - all_grids["Combined"][1][:-1]
    assert np.isclose(dy, dy[0]).all()


    plt.figure(2)
    
    for cut_val in np.linspace(6.15, 6.21, 100):
        included_points = float((all_grids["Combined"][2] <= cut_val).sum())
        included_area = included_points*dx[0]*dy[0]

        plt.plot(cut_val, 1./included_area, '.', color = 'b')
    plt.axvline(6.18007)
    plt.savefig("DETF_evaluation.pdf")
    plt.close()
        
    included_points = float((all_grids["Combined"][2] <= 6.18007).sum())
    included_area = included_points*dx[0]*dy[0]
    
    DETF_FoM = 1./included_area
    return DETF_FoM

def make_contours(all_grids, BAO_Omh2):
    DETF_FoM_txt = ""
    
    if all_grids["model"] == "flatwCDM":
        plt.figure(figsize = (5,5))
    elif (all_grids["model"] == "flatw0wa") or (all_grids["model"] == "w0wa"):
        plt.figure(figsize = (5,5))
        DETF_FoM = get_DETF(all_grids)
        DETF_FoM_txt = "\nDETF FoM: %.2f" % DETF_FoM

        
    elif all_grids["model"] == "LCDM":
        plt.figure(figsize = (5,7.5))
    else:
        assert 0


    if "BAO" in all_grids:
        BAO_key = "BAO" + "_Omh2"*BAO_Omh2
        plt.contourf(all_grids[BAO_key][0], all_grids[BAO_key][1], all_grids[BAO_key][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("green"))
        plt.contourf(all_grids["CMB"][0], all_grids["CMB"][1], all_grids["CMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("orange"))
        plt.contourf(all_grids["SNe"][0], all_grids["SNe"][1], all_grids["SNe"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("blue"))
        plt.contourf(all_grids["Combined"][0], all_grids["Combined"][1], all_grids["Combined"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("gray"))
        
        
        plt.contour(all_grids["SNe"][0], all_grids["SNe"][1], all_grids["SNe"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25)
        plt.contour(all_grids["CMB"][0], all_grids["CMB"][1], all_grids["CMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25, linestyles = "dotted")
        plt.contour(all_grids[BAO_key][0], all_grids[BAO_key][1], all_grids[BAO_key][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25, linestyles = "dashed")
    else:
        if BAO_Omh2:
            return 0
        
        plt.contourf(all_grids["Combined"][0], all_grids["Combined"][1], all_grids["Combined"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("teal"))
        plt.contour(all_grids["Combined"][0], all_grids["Combined"][1], all_grids["Combined"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25)

        

    if all_grids["model"] == "flatwCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$w$")
        plt_name = "Om-w%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)

    elif all_grids["model"] == "flatw0wa":
        plt.xlabel("$w_0$")
        plt.ylabel("$w_a$")
        plt_name = "w0-wa%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)

    elif all_grids["model"] == "w0wa":
        plt.xlabel("$w_0$")
        plt.ylabel("$w_a$")
        plt_name = "w0-wa_open%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)

    elif all_grids["model"] == "LCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$\Omega_{\Lambda}$")
        plt.plot([0, 1], [1, 0], color = 'k', linewidth = 0.75)
        plt.axes().set_aspect(1.)
        plt.xlim(0, 1)
        plt.ylim(0, 1.5)
        plt_name = "Om-OL%s.pdf" % ("_BAO_Omh2"*BAO_Omh2)

    else:
        assert 0

    all_txt = "All: " + str(all_grids["Combined_fit"]) + " " + str(np.sqrt(np.diag(all_grids["Combined_cmat"]))) + '\n'
    all_txt += "SN+CMB: " + str(all_grids["SNCMB_fit"]) + " " + str(np.sqrt(np.diag(all_grids["SNCMB_cmat"]))) + '\n'
    all_txt += "BAO+CMB: " + str(all_grids["BAOCMB_fit"]) + " " + str(np.sqrt(np.diag(all_grids["BAOCMB_cmat"])))
    all_txt += DETF_FoM_txt
    
    plt.savefig(plt_name, bbox_inches = 'tight', metadata=dict(Keywords = all_txt))

    

for fl in sys.argv[1:]:
    all_grids = pickle.load(open(fl, 'rb'))
    for key in all_grids:
        print(key)
    print("Combined_minos", all_grids["Combined_minos"])
    make_contours(all_grids, BAO_Omh2 = 0)
    make_contours(all_grids, BAO_Omh2 = 1)

