from FileRead import readcol
import numpy as np
import sys
import tqdm
from scipy.special import erfinv

def twoD_chi2_to_sigma(x):
    return np.sqrt(2) * erfinv(1 - np.exp(-x / 2.0))


burn = 0.3

for item in sys.argv[1:]:
    [weight, minuslogpost, cosmomc_theta, ombh2, omch2, tau, ns, logA, w, wa] = readcol(item, 'i,f,f,ff,f,f,f,ff')
    
    print("weight", set(weight))
    
    full_w = []
    full_wa = []
    
    for samp in tqdm.trange(len(weight)):
        for j in range(weight[samp]):
            full_w.append(w[samp])
            full_wa.append(wa[samp])
        
    print(len(full_w))
    print(len(full_wa))

    min_sample = int(len(full_w)*burn)

    full_w = full_w[min_sample:]
    full_wa = full_wa[min_sample:]

    print(len(full_w))
    print(len(full_wa))
    
    samps = np.array([full_w, full_wa])
    cmat = np.cov(samps)
    print(cmat, np.sqrt(np.diag(cmat)))
    
    wmat = np.linalg.inv(cmat)
    med = np.median(samps, axis = 1)
    print(med)
    
    resid = np.array([-1., 0.]) - med
    print("resid", resid)
    
    chi2_LCDM = np.dot(resid, np.dot(wmat, resid))
    print("chi2 of -1 0 ", item, chi2_LCDM, " sigma: ", twoD_chi2_to_sigma(chi2_LCDM))

