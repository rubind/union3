import pickle
import numpy as np
import glob
import gzip
import sys


#sample_fl = glob.glob("samples*pickle")
#assert len(sample_fl) == 1
#sample_fl = sample_fl[0]

sample_fl = sys.argv[1]

input_fl = glob.glob("inputs*pickle")
assert len(input_fl) == 1
input_fl = input_fl[0]




try:
    fit_params = pickle.load(open(sample_fl, 'rb'))
except:
    fit_params = pickle.load(gzip.open(sample_fl, 'rb'))

try:
    (the_data, stan_data, params) = pickle.load(open(input_fl, 'rb'))
except:
    (the_data, stan_data, params) = pickle.load(gzip.open(input_fl, 'rb'))


assert len(the_data["calib_names"]) == len(fit_params["calibs"][0])

for i in range(len(the_data["calib_names"])):
    the_med = np.median(fit_params["calibs"][:,i])
    the_rms = np.std(fit_params["calibs"][:,i], ddof=1)
    if np.abs(the_med) > 0.25:
        print("the_med", the_med, "posterior RMS", the_rms, the_data["calib_names"][i])
