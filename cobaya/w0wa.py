from FileRead import readcol
import numpy as np
import sys
import tqdm
from scipy.special import erfinv
from DavidsNM import miniNM_new
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
from sklearn.mixture import GaussianMixture

def log_density_ratio_gmm(samples, theta_peak, theta_test, K=2, seed=0):
    """
    samples: (N,2) posterior samples
    theta_peak: (2,) point near mode
    theta_test: (2,) point (e.g. ~3 sigma excluded)
    returns: Δ = log p(peak) - log p(test)
    """
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        reg_covar=1e-8,          # tiny regularization helps stability in tails
        n_init=5,                # reinit EM a few times
        random_state=seed
    )
    gmm.fit(samples)

    lp_peak = gmm.score_samples(np.atleast_2d(theta_peak))[0]
    lp_test = gmm.score_samples(np.atleast_2d(theta_test))[0]

    return lp_peak - lp_test


# Example usage:
# samples = ... shape (N,2)
# theta_peak = np.array([0.1, -0.05])
# theta_test = np.array([1.2, 0.7])




def twoD_chi2_to_sigma(x):
    return np.sqrt(2) * erfinv(1 - np.exp(-x / 2.0))


burn = 0.3

n_gauss = int(sys.argv[1])

for item in sys.argv[2:]:
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
        
    print("Expanded repeats:")
    print(len(full_w))
    print(len(full_wa))

    min_sample = int(len(full_w)*burn)

    full_w = full_w[min_sample:]
    full_wa = full_wa[min_sample:]

    print("After burn:")
    print(len(full_w))
    print(len(full_wa))
    
    samps = np.array([full_w, full_wa])

    plt.plot(full_w, full_wa, '.', alpha = 0.005, color = 'b')
    wa_percentiles = scoreatpercentile(full_wa, [15.8655, 50., 84.1345])
    plt.title("%f +%f -%f" % (wa_percentiles[1], wa_percentiles[2] - wa_percentiles[1], wa_percentiles[1] - wa_percentiles[0]))
    plt.savefig("w0wa.pdf", bbox_inches = 'tight')
    plt.close()
    
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



    for K in [1, 2, 3]:
        dlogp = log_density_ratio_gmm(samps.T, med, theta_test = [-1., 0.], K=K)
        print(f"K={K:2d}: log(p_peak/p_test) = {dlogp:.4f}, 2log(p_peak/p_test) = {2*dlogp:.4f},  ratio = {np.exp(dlogp):.3e}")
        print(" sigma: ", twoD_chi2_to_sigma(2*dlogp))
