from FileRead import readcol
import numpy as np
import sys
import tqdm
from scipy.special import erfinv

def twoD_chi2_to_sigma(x):
    return np.sqrt(2) * erfinv(1 - np.exp(-x / 2.0))


burn = 0.3

for item in sys.argv[1:]:
    f = open(item, 'r')
    lines = f.read().split('\n')
    f.close()

    weight_ind = lines[0].split(None).index("weight") - 1
    w_ind = lines[0].split(None).index("w") -	1
    wa_ind = lines[0].split(None).index("wa") -   1

    assert weight_ind == 0
    assert w_ind < 10 and w_ind > 0
    assert wa_ind < 10 and wa_ind > 0

    first_cols = readcol(item, 'i,ffffffffff')

    weight = first_cols[weight_ind]
    w = first_cols[w_ind]
    wa = first_cols[wa_ind]
    
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
    av = np.mean(samps, axis = 1)
    print(med)
    print(av)

    for cent in [med, av]:
        resid = np.array([-1., 0.]) - cent
        print("resid", resid)
        
        chi2_LCDM = np.dot(resid, np.dot(wmat, resid))
        print("chi2 of -1 0 ", item, chi2_LCDM, " sigma: ", twoD_chi2_to_sigma(chi2_LCDM))

