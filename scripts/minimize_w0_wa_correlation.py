from DavidsNM import miniNM_new
import numpy as np
import pickle
import gzip
import sys

def chi2fn(P, passdata):
    [w0, wa] = passdata[0]

    wp = w0 + wa*P[0]

    dwp = wp - np.mean(wp)
    dwa = wa - np.mean(wa)

    cross_term = sum(dwp*dwa)
    return cross_term**2.


all_vals = []


for pfl in sys.argv[1:]:
    fit_params = pickle.load(gzip.open(pfl, 'rb'))

    P, NA, NA = miniNM_new(ministart = [0.0], miniscale = [1.], chi2fn = chi2fn, passdata = [fit_params["wDE"], fit_params["waDE"]])

    all_vals.append(P[0])

print("all_vals", all_vals, np.mean(all_vals), np.median(all_vals))

