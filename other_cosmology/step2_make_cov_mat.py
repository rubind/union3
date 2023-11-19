from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cosmo_functions
import sys


samps = ascii.read(sys.argv[1])

columns = list(samps.columns)
columns.sort()
print(columns)

if columns.count("omegak") and columns.count("w"):
    assert 0

plt.figure(figsize = (24, 16))

R_theta_Obhh = []
R_theta_Obhh_smallk = []

all_samps_for_comparison = {1: [], 2: [], 5: [], 7: []}


for i in tqdm.trange(1, len(samps)):
    if columns.count("omegak"):
        this_cosmo = dict(O_m = samps["omegam*"][i], h = samps["H0*"][i]/100., O_bhh = samps["omegabh2"][i], model = "LCDM", O_k = samps["omegak"][i])
        this_O_k = samps["omegak"][i]
        this_w = np.random.normal()
    elif columns.count("w"):
        this_cosmo = dict(O_m = samps["omegam*"][i], h = samps["H0*"][i]/100., O_bhh = samps["omegabh2"][i], model = "flatwCDM", O_k = 0, w = samps["w"][i])
        this_O_k = np.random.normal()
        this_w = samps["w"][i]

    z_star_mine = cosmo_functions.getzstar(O_m = samps["omegam*"][i], hh = (samps["H0*"][i]/100.)**2., O_bhh = samps["omegabh2"][i])
    r_star_mine = cosmo_functions.get_sound_horizon(cosmo = this_cosmo, CMB_not_BAO = 1)
    r_star_mine *= cosmo_functions.CosConst.c100_Mpc / (samps["H0*"][i]/100.)

    r_drag_mine = cosmo_functions.get_sound_horizon(cosmo = this_cosmo, CMB_not_BAO = 0)
    r_drag_mine *= cosmo_functions.CosConst.c100_Mpc / (samps["H0*"][i]/100.)

    theta_mine = 100 * r_star_mine/ cosmo_functions.highz_r(cosmo = this_cosmo, zmin = 0., zmax = z_star_mine)
    theta_mine /= cosmo_functions.CosConst.c100_Mpc / (samps["H0*"][i]/100.)
    
    r_drag_samp = samps["rdrag*"][i]
    
    z_star_samp = samps["zstar*"][i]
    r_star_samp = samps["rstar*"][i]



    all_samps_for_comparison[1].append(np.log(z_star_samp/z_star_mine))
    all_samps_for_comparison[2].append(np.log(r_star_samp/r_star_mine))
    all_samps_for_comparison[5].append(np.log(r_drag_samp/r_drag_mine))
    all_samps_for_comparison[7].append(np.log(samps["theta"][i]/theta_mine))
    
    
    
    if np.random.random() < 0.01:
        plt.subplot(3,3,1)
        pltvals = np.log(z_star_samp/z_star_mine)
        plt.plot(z_star_samp, pltvals, '.', color = 'b')
        plt.xlabel("z_star_samp")
        plt.ylabel("log(z_star_samp/z_star_mine)")
        plt.axhline(0)

        plt.subplot(3,3,2)
        pltvals = np.log(r_star_samp/r_star_mine)
        plt.plot(r_star_samp, pltvals, '.', color = 'b')
        plt.xlabel("r_star_samp")
        plt.ylabel("log(r_star_samp/r_star_mine)")
        plt.axhline(0)

        plt.subplot(3,3,3)
        if columns.count("omegak"):
            plt.plot(samps["omegam*"][i], samps["omegak"][i], '.', color = 'b')
            plt.xlabel("omegam")
            plt.ylabel("omegak")
        else:
            plt.plot(samps["omegam*"][i], samps["w"][i], '.', color = 'b')
            plt.xlabel("omegam")
            plt.ylabel("w")

        plt.subplot(3,3,4)
        plt.plot(z_star_samp, samps["omegabh2"][i], '.', color = 'b')
        plt.xlabel("z_star_samp")
        plt.ylabel("omegabh2")

        plt.subplot(3,3,5)
        pltvals = np.log(r_drag_samp/r_drag_mine)
        plt.plot(r_drag_samp, pltvals, '.', color = 'b')
        plt.xlabel("r_drag_samp")
        plt.ylabel("log(r_drag_samp/r_drag_mine)")
        plt.axhline(0)

        plt.subplot(3,3,6)
        plt.plot(samps["theta"][i], np.log(samps["theta"][i]/samps["thetastar*"][i]), '.', color = 'b')
        plt.xlabel("theta")
        plt.ylabel("log(theta/thetastar)")
        plt.axhline(0)

        plt.subplot(3,3,7)
        pltvals = np.log(samps["theta"][i]/theta_mine)
        plt.plot(samps["theta"][i], pltvals, '.', color = 'b')
        plt.xlabel("theta")
        plt.ylabel("log(theta/theta_mine)")
        plt.axhline(0)

    R_mine = cosmo_functions.get_R(this_cosmo)
    
    R_theta_Obhh.append([R_mine, theta_mine, samps["omegabh2"][i], samps["omegamh2*"][i], r_star_mine, this_O_k, this_w])
    if np.abs(this_O_k) < 0.01:
        R_theta_Obhh_smallk.append(R_theta_Obhh[-1])

for key in all_samps_for_comparison:
    plt.subplot(3,3,key)
    plt.title("Mean: %.4f RMS: %.2g" % (np.mean(all_samps_for_comparison[key]),
                                        np.std(all_samps_for_comparison[key], ddof=1)))
        
        
plt.savefig("planck_chain_" + sys.argv[1].split(".")[0] + ".pdf", bbox_inches = 'tight')

R_theta_Obhh = np.array(R_theta_Obhh).T
R_theta_Obhh = np.array([R_theta_Obhh])

R_theta_Obhh_smallk = np.array(R_theta_Obhh_smallk).T
R_theta_Obhh_smallk = np.array([R_theta_Obhh_smallk])

import kde_corner

Obhh_Omhh = R_theta_Obhh[0][2:4]
print("Obhh Omhh", np.median(Obhh_Omhh, axis = 1))

Obhh_Omhh_cmat = np.cov(Obhh_Omhh)

print("Obhh Omhh", Obhh_Omhh_cmat)
print("Obhh Omhh", np.linalg.inv(Obhh_Omhh_cmat))




kde_corner.kde_corner(R_theta_Obhh, labels = ["R", "theta", "Obh^2", "omegamh2", "r_star_mine", "O_k", "w"])
plt.savefig("CMB_corner" + sys.argv[1].split(".")[0] + ".pdf", bbox_inches = 'tight')
plt.close()




c_mat = np.cov(R_theta_Obhh[0])
print("c_mat", c_mat.shape)

med_vals = np.median(R_theta_Obhh[0], axis = 1)
print("med_vals", med_vals[:3])

merged = np.concatenate((c_mat[:3,:3], np.linalg.inv(c_mat[:3,:3]), [med_vals[:3]]))
from DavidsNM import save_img
save_img(merged, "merged_vals_" + sys.argv[1].split(".")[0] + ".fits")

to_print = [str(item) for item in med_vals[:3]]
print("Values & " + " & ".join(to_print) + "\\\\")

tmp_wmat = np.linalg.inv(c_mat[:3,:3])

tmp_wmat = 0.5*(tmp_wmat + tmp_wmat.T) # Force symmetric

to_print = ["$" + str(item) + "$" for item in tmp_wmat[0]]
print("$R$ & " + " & ".join(to_print) + "\\\\")
to_print = ["$" + str(item) + "$" for item in tmp_wmat[1]]
print("$l_A$ & " + " & ".join(to_print) + "\\\\")
to_print = ["$" + str(item) + "$" for item in tmp_wmat[2]]
print("$\omega_b$ & " + " & ".join(to_print) + "\\\\")


if columns.count("omegak"):
    kde_corner.kde_corner(R_theta_Obhh_smallk, labels = ["R", "l_A", "Obh^2", "omegamh2", "r_star_mine", "O_k", "w"])
    plt.savefig("CMB_corner_smallk" + sys.argv[1].split(".")[0] + ".pdf", bbox_inches = 'tight')
    plt.close()
