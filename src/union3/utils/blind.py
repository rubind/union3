import os
from scipy.interpolate import interp1d
import numpy as np

from union3.utils.file_read import readcol


def blind(stan_data, the_data, params):
    print("Blinding!")
    # Blind H0

    [zblind, mublind, NA] = readcol(params["blinding_fl"].replace("$UNITY", os.environ["UNITY"]), "fff")
    mublindfn = interp1d(zblind, mublind, kind="linear")
    # dmublinddOmfn = interp1d(zblind, dmublinddOm, kind = 'linear')

    target_distmod = mublindfn(stan_data["redshifts"])
    inds = np.where(stan_data["distmod"] > 0)
    med_offset = np.median(target_distmod[inds] - stan_data["distmod"][inds])
    stan_data["distmod"] += med_offset

    # There are two phases of Hubble-flow blinding:
    # -Making the best-fit Om = 0.3
    # -Bringing all samples into alignment with -19.1 given Om = 0.3

    for iter_count in range(2):
        muobs = (
            stan_data["obs_mBx1c"][:, 0]
            + 0.14 * stan_data["obs_mBx1c"][:, 1]
            - 3.1 * stan_data["obs_mBx1c"][:, 2]
            - -19.1
        )
        H_resid = muobs - mublindfn(stan_data["redshifts"])

        for sample_ind in range(stan_data["n_samples"]):
            inds = np.where((the_data["sample_list"] == sample_ind) * (stan_data["redshifts"] >= 0.01))

            if len(inds[0]) > 0:
                med_HR = np.median(H_resid[inds])

                inds = np.where((the_data["sample_list"] == sample_ind))

                for SN_ind in inds[0]:
                    stan_data["obs_mBx1c"][SN_ind, 0] -= med_HR
                    the_data["mB_list"][SN_ind] -= med_HR

                if iter_count > 0:
                    assert abs(med_HR) < 1e-3

    print("Blinding complete!")
    return stan_data, the_data
