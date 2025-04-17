from FileRead import readcol
import numpy as np
import sys
import glob
from DavidsNM import save_img, miniLM_new

def residfn(P, passdata):
    mod = np.outer(P, P)
    resid = passdata[0] - mod
    for dind in range(len(resid)):
        resid[dind, dind] = 0

    return np.concatenate((
        (resid/0.000001).flatten(),
        P)) # Weak prior on P to eliminate one large element and the rest zeros

uniondir = sys.argv[1]

[mu_vals] = readcol("only_cephs/r22_mcmc_muVals.dat", 'f')
[host_names] = readcol("only_cephs/r22_mcmc_Hosts.txt", 'a')
[hostSN1, hostSN2] = readcol("only_cephs/r22_mcmc_hostSNmap.csv", 'aa', splitchar = ",")

if hostSN1[0] == "host":
    hostSN1 = hostSN1[1:]
    hostSN2 = hostSN2[1:]
    
mu_cov = np.loadtxt("only_cephs/r22_mcmc_muCov.dat")




assert len(mu_vals) == len(host_names)
assert len(mu_cov) == len(mu_vals)

assert (mu_cov.T == mu_cov).all()

save_img(mu_cov, "only_cephs/mu_cov.fits")


host_inds = []
all_drs = []

for SN_ind in range(len(hostSN1)):
    host_ind = host_names.index(hostSN1[SN_ind])

    drs = glob.glob(uniondir + "/*/*" + hostSN2[SN_ind])
    assert len(drs) < 2, str(drs)
    if len(drs) == 1:
        dr = drs[0]
        print("Matched", dr, hostSN1[SN_ind], hostSN2[SN_ind])

        host_inds.append(host_ind)
        all_drs.append(dr)
        
    else:
        print("Missing", hostSN1[SN_ind], hostSN2[SN_ind])

print("host_inds", host_inds)

full_mu_cov = np.zeros([len(host_inds)]*2, dtype=np.float64)

for i in range(len(host_inds)):
    for j in range(len(host_inds)):
        full_mu_cov[i,j] = mu_cov[host_inds[i], host_inds[j]]

save_img(full_mu_cov, "only_cephs/full_mu_cov.fits")


remaining_mu_cov = full_mu_cov * 1.

all_evs = []

done = 0
while done == 0:
    P, F, NA = miniLM_new(ministart = np.random.random(size = len(full_mu_cov))*0.01, miniscale = np.ones(len(full_mu_cov), dtype=np.float64),
                           residfn = residfn, passdata = remaining_mu_cov, verbose = False)
    print("Iter ", i, F)

    if np.all(np.diag(remaining_mu_cov) > P**2):
        remaining_mu_cov -= np.outer(P, P)

        save_img(remaining_mu_cov, "only_cephs/mu_cov_remain_%02i.fits" % i)
        all_evs.append(P)
    else:
        done = 1
        

save_img(all_evs, "only_cephs/all_evs.fits")

assert np.all(np.diag(remaining_mu_cov) > 0)

for full_matrix in [0, 1]:
    f = open("dist_ladder_R22" + "_full_matrix"*full_matrix + ".txt", 'w')
    
    assert len(all_drs) == len(remaining_mu_cov)
    assert len(all_drs) == len(all_evs[0])
    assert len(all_drs) == len(host_inds)
    
    for SN_ind in range(len(all_drs)):
        f.write(all_drs[SN_ind].split("/")[-1] + "  " + str(mu_vals[host_inds[SN_ind]]) + "  " + str(np.sqrt(remaining_mu_cov[SN_ind, SN_ind])*(1. - full_matrix)) + "  ")

        for ev in all_evs:
            f.write(str(ev[SN_ind]) + "  ")

        if full_matrix:
            for SN_ind2 in range(len(all_drs)):
                if SN_ind2 == SN_ind:
                    f.write(str(np.sqrt(remaining_mu_cov[SN_ind, SN_ind])) + "  ")
                else:
                    f.write("0.0  ")
                    
        f.write('\n')
        

    f.close()
