import numpy as np
import matplotlib.pyplot as plt
import pystan
import pickle


stan_code = """
// Version History. Update in print statement in transformed data!
// Version 1.5. First official release with new selection-effect and population model!



data {
    int<lower=0> n_sne; // number of SNe
    int <lower = 0, upper = 1> threeD_unexplained;

    vector[2] obs_mBc [n_sne];
    matrix[2,2] obs_mBc_cov [n_sne];
}


parameters {
    real MB;
    real <lower = -1.4, upper = 1.4> beta_angle;
    real <lower=0.01, upper = 0.3> sigma_int;
    simplex [2] mBc_int_variance;

    vector [n_sne] true_c;


    // Population parameters:
  
    real c_star;
    real <lower= 0, upper = 0.2> R_c;
    real <lower = -1, upper = 1> mobs_cut;
    real <lower = 0.1> mobs_cut_sigma;
}

transformed parameters {
    vector [2] model_mBc [n_sne];
    matrix [2,2] model_mBc_cov [n_sne];

    vector [2] sig_int_vector;

    real beta;

    vector [n_sne] mobs_by_SN;
    vector [n_sne] mobs_var_by_SN;
    real this_norm_LL;
    vector [n_sne] inl_loglike_by_SN;

    model_mBc_cov <- obs_mBc_cov;
    

    beta <- tan(beta_angle);
    


    if (threeD_unexplained == 1) {
        sig_int_vector[1] <- sqrt(mBc_int_variance[1])*sigma_int;        // This vector is in dispersion, not variance
        sig_int_vector[2] <- sqrt(mBc_int_variance[2])*sigma_int/(-3.);
    } else {
        sig_int_vector[1] <- sigma_int;        // This vector is in dispersion, not variance
        sig_int_vector[2] <- 0.;
    }
    

    for (i in 1:n_sne) {
        for (j in 1:2) {
            model_mBc_cov[i][j,j] <- model_mBc_cov[i][j,j] + sig_int_vector[j]^2;
        }

	mobs_by_SN[i] <- MB + beta*c_star;
	mobs_var_by_SN[i] <- mobs_cut_sigma^2 + model_mBc_cov[i][1,1] + (beta*R_c)^2;

        model_mBc[i][1] <- MB + beta*true_c[i];
        model_mBc[i][2] <- true_c[i];

        this_norm_LL = 0.0001;
	this_norm_LL += normal_cdf(   mobs_cut,
                                      mobs_by_SN[i], sqrt(mobs_var_by_SN[i] ));

	inl_loglike_by_SN[i] <- multi_normal_log(obs_mBc[i], model_mBc[i], model_mBc_cov[i])
					  + normal_log(true_c[i], c_star, R_c)

	                                  + normal_cdf_log(mobs_cut,  obs_mBc[i][1], mobs_cut_sigma)
                                          - log(this_norm_LL);  //No calibration in this term, see above comment!
    }

}

model {

    for (i in 1:n_sne) {
        target += inl_loglike_by_SN[i];
    }

    mobs_cut ~ normal(0., 0.5);
    mobs_cut_sigma ~ normal(0.5, 0.25);


    c_star  ~ normal(0.1, 0.2);
    R_c ~ normal(0.1, 0.2);
}
"""

n_sne = 4000
true_c = np.random.normal(size = n_sne*10)*0.1
obs_mB = true_c*3. + np.random.normal(size = n_sne*10)*0.1
obs_c = true_c + np.random.normal(size = n_sne*10)*0.04

inds = np.where(obs_mB + np.random.normal(size = n_sne*10)*0.15 < 0)
true_c = true_c[inds][:n_sne]
obs_mB = obs_mB[inds][:n_sne]
obs_c = obs_c[inds][:n_sne]

stan_data = dict(n_sne = n_sne, 
                 threeD_unexplained = 1,
                 obs_mBc = [[obs_mB[i], obs_c[i]] for i in range(n_sne)],
                 obs_mBc_cov = [np.array([[0.04**2., 0.],
                                            [0.,    0.04**2.]]) for i in range(n_sne)])

sm = pystan.StanModel(model_code=stan_code)
fit = sm.sampling(data=stan_data, iter=2000, chains=4, refresh = 10)
print(fit.stansummary(digits_summary=5))

