import sys
import pickle
import numpy as np
import gzip
import tqdm
from astropy.io import fits
from DavidsNM import save_img
import glob


max_params_to_save = int(sys.argv[1])
thin_by = int(sys.argv[2])
common_path = sys.argv[3]

all_samples = {}

save_mu_mat = len(glob.glob("/".join(sys.argv[3].split("/")[:-1]) + "/mu_mat.fits")) > 0


if save_mu_mat:
    f = fits.open("/".join(sys.argv[3].split("/")[:-1]) + "/mu_mat.fits")
    dat = f[0].data
    f.close()


for pfl in tqdm.tqdm(sys.argv[3:]):
    ind = 0
    while (ind < len(common_path)) and (pfl[ind] == common_path[ind]):
        ind += 1

    common_path = common_path[:ind]

    fit_params = pickle.load(gzip.open(pfl, 'rb'))

    del_keys = []
    for key in fit_params:
        sh = np.array(fit_params[key].shape)

        if np.any(sh[1:] > max_params_to_save):
            print(key, " is too big to save!", sh)
            del_keys.append(key)

    print("del_keys", del_keys)
    for key in del_keys:
        del fit_params[key]

        
    for key in fit_params:
        if thin_by > 1:
            print("thinning...")
            fit_params[key] = fit_params[key][::thin_by]

        
        if key in all_samples:
            all_samples[key] = np.concatenate((all_samples[key], fit_params[key]))
        else:
            all_samples[key] = fit_params[key]

    if save_mu_mat:
        f = fits.open("/".join(pfl.split("/")[:-1]) + "/mu_mat.fits")
        this_dat = f[0].data
        f.close()
        
        assert np.all(this_dat[0] == dat[0])

            
    
print("common_path", common_path)
while common_path[-1] == "_":
    common_path = common_path[:-1]
print("common_path", common_path)

if thin_by > 1:
    common_path += ("_thin=%i" % thin_by)

pickle.dump(all_samples, gzip.open("all_samples_" + common_path + ".pickle", "wb"))

if save_mu_mat:
    mu_cov = np.cov(all_samples["mu_zbins"].T)
    
    dat[1:, 1:] = np.linalg.inv(mu_cov)
    dat[1:, 0] = np.median(all_samples["mu_zbins"], axis = 0)
    
    save_img(dat, "mu_mat_" + common_path + ".fits")
