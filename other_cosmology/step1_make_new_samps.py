from FileRead import readcol, writecol
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import tqdm


# As of 2023-Jan-06:
# ~/Downloads/COM_CosmoParams_fullGrid_R3/base_w/plikHM_TTTEEE_lowl_lowE/base_w_plikHM_TTTEEE_lowl_lowE
# ~/Downloads/COM_CosmoParams_fullGrid_R3/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE


prefix = sys.argv[1]


[param_names] = readcol(prefix + ".paramnames", 'a')

all_samps = np.zeros([len(param_names) + 2, 0], dtype=np.float64)

for fl in glob.glob(prefix + "_?.txt"):
    these_samps = np.loadtxt(fl).T
    all_samps = np.concatenate((all_samps, these_samps), axis = 1)

print("all_samps", all_samps.shape)

new_samps = np.zeros([len(param_names) + 2, 0], dtype=np.float64)
for i in tqdm.trange(len(all_samps[0])):
    new_samps = np.concatenate((new_samps, (all_samps[:,i]*np.ones([int(all_samps[0,i]), len(param_names) + 2], dtype=np.float64)).T), axis = 1)

print("new_samps", new_samps.shape)
        
plt.hist(new_samps[0], bins = 50)
plt.hist(all_samps[0], bins = 50)
plt.savefig("weights.pdf")
plt.close()

writecol("new_samps_" + prefix.split("/")[-1] + ".txt", new_samps, headings = ["count", "LL"] + param_names, doformat = False)
