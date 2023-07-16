import pickle as pickle
import numpy as np
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile, percentileofscore
import sys
import subprocess
import gzip
from FileRead import readcol
import time
import os
from DavidsNM import miniNM_new, save_img, miniLM_new
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import subprocess
from scipy.interpolate import interp1d, interpnd, NearestNDInterpolator

def residfn(P, passdata):
    [dmags, eigenvectors, inds, prior] = passdata[0]

    mag_model = np.dot(eigenvectors.T, P)[inds]
    return np.concatenate(((dmags - mag_model)/0.15, P/prior))


def test_for_flow(eigenvectors, z_CMB):
    inds = np.where(z_CMB < zmax)
    z_CMB = z_CMB[inds]
    RA = np.array(np.array(the_data["RA"])[inds], dtype=np.float64)
    Dec = np.array(np.array(the_data["Dec"])[inds], dtype=np.float64)

    for flowdir in "xyz":
        if flowdir == "x":
            dmags = (0.002/z_CMB) * np.cos(RA/57.3) * np.cos(Dec/57.3)
        elif flowdir == "y":
            dmags = (0.002/z_CMB) * np.sin(RA/57.3) * np.cos(Dec/57.3)
        else:
            dmags = (0.002/z_CMB) * np.sin(Dec/57.3)
            
        for prior in [0.1, 1., 100.]:
            P, NA, NA = miniLM_new(ministart = np.random.normal(size = 100),
                                   miniscale = np.ones(100, dtype=np.float64),
                                   residfn = residfn, passdata = [dmags, eigenvectors, inds, prior], verbose=True)

            mag_model = np.dot(eigenvectors.T, P)[inds]

            plt.figure(figsize = (18, 6))
            plt.subplot(1,2,1)
            plt.scatter(dmags, mag_model)
            plt.title("RMS resid.: %.3f" % np.std(dmags - mag_model, ddof=1))
            plt.subplot(1,2,2)
            plt.hist(P, bins = 20)
            plt.savefig("mag_model_vs_dmags_prior=%.1g_flow=%s.pdf" % (prior, flowdir))
            plt.close()


input_fl = sys.argv[1]
sample_fl = sys.argv[2]
zmax = float(sys.argv[3]) # typically 0.05


(the_data, stan_data, params) = pickle.load(gzip.open(input_fl, 'rb'))
fit_params = pickle.load(gzip.open(sample_fl, 'rb'))


for key in the_data:
    print("the_data", key)
    
bulk_samples = []

print("d_mBx1c_dcalib_list", stan_data["d_mBx1c_d_calib"].shape)


assert len(the_data["calib_names"]) == len(fit_params["calibs"][0])

eigenvectors = []

for i in range(len(the_data["calib_names"])):
    if the_data["calib_names"][i].count("BULK"):
        assert the_data["calib_names"][i] == "BULK_%03i" % len(bulk_samples)
        bulk_samples.append(fit_params["calibs"][:,i])
        eigenvectors.append(stan_data["d_mBx1c_d_calib"][:,0,i])
        assert np.all(stan_data["d_mBx1c_d_calib"][:,1:,i] == 0)

        
bulk_samples = np.array(bulk_samples)
bulk_med = np.median(bulk_samples, axis = 1)
bulk_cov = np.cov(bulk_samples)

print("bulk_cov", bulk_cov.shape)

save_img(bulk_cov, "bulk_cov.fits")

print("bulk_samples", bulk_samples.shape)
eigenvectors = np.array(eigenvectors)
print("eigenvectors", eigenvectors.shape)



z_CMB = np.array(the_data["z_CMB_list"])

test_for_flow(eigenvectors, z_CMB)
ffff

inds = np.where(z_CMB < zmax)
z_CMB = z_CMB[inds]
RA = np.array(np.array(the_data["RA"])[inds], dtype=np.float64)
Dec = np.array(np.array(the_data["Dec"])[inds], dtype=np.float64)

xs = z_CMB*np.cos(RA/57.3)*np.cos(Dec/57.3)
ys = z_CMB*np.sin(RA/57.3)*np.cos(Dec/57.3)
zs = z_CMB*np.sin(Dec/57.3)


this_real = bulk_med*1.
mag_med = np.dot(eigenvectors.T, this_real)
vel_med = mag_med[inds]*z_CMB*299792.458/(1. + z_CMB) * np.log(10.)/5.


print("vel_med", vel_med)


ifn = NearestNDInterpolator(np.array([xs, ys, zs]).T, vel_med)

pltxs = np.linspace(-zmax, zmax, 500)
pltys = np.linspace(-zmax, zmax, 501)

all_plt_z = np.zeros([len(pltxs), len(pltys)], dtype=np.float64)
for i in tqdm.trange(len(pltxs)):
    for j in range(len(pltys)):
        all_plt_z[i, j] = ifn(pltxs[i], pltys[j], 0.)

plt.imshow(all_plt_z)
plt.colorbar()
plt.savefig("2d_flow.pdf", bbox_inches = 'tight')
plt.close()

fffff

random_reals = np.random.multivariate_normal(mean = bulk_med, cov = bulk_cov, size = 30)
print("random_reals", random_reals.shape)

angle_nodes = np.linspace(0., 360., len(random_reals) + 1)

ifns = []
for i in range(len(random_reals[0])):
    ifns.append(interp1d(angle_nodes, np.concatenate((random_reals[:,i], [random_reals[9,i]])), kind = 'linear'))


angles = np.linspace(0., 360., 300 + 1)[:-1]

subprocess.getoutput("rm -fr flow_frames")
subprocess.getoutput("mkdir flow_frames")

for i in tqdm.trange(len(angles)):
    this_real = np.zeros(len(eigenvectors), dtype=np.float64)
    for j in range(len(eigenvectors)):
        this_real[j] = ifns[j](angles[i])
        
    mag_med = np.dot(eigenvectors.T, this_real)
    vel_med = mag_med[inds]*z_CMB*299792.458/(1. + z_CMB) * np.log(10.)/5.

    
    ax = plt.subplot(1,1,1, projection='3d')
    ax.view_init(30, angles[i])

    vals = ax.scatter(xs, ys, zs, c = vel_med, s = 2, cmap = 'rainbow', vmin = -250, vmax = 250)
    plt.colorbar(vals)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.savefig("flow_frames/frame_%04i.png" % i)
    plt.close()

subprocess.getoutput("convert -delay 5 flow_frames/* all_frames.gif")
