import gzip
import cPickle as pickle
from numpy import *
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile

pfls = ["orig/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_KC_v1_CfA3_4S_v1_CfA4_v1.pickle",
        "meanstar/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_KC_v1_CfA3_4S_v1_CfA4_v1.pickle",
        "meanstarX/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_KC_v1_CfA3_4S_v1_CfA4_v1.pickle"]

colors = 'rgb'

for i in range(3):
    fit_params = pickle.load(gzip.open(pfls[i], 'rb'))

    MB_high_mass = fit_params["MB"] - outer(fit_params["delta_0"], ones(6))

    plt.plot(arange(6) + 0.1*i, median(MB_high_mass, axis = 0), 'o', label = pfls[i].split("/")[0], color = colors[i])
    p16 = scoreatpercentile(MB_high_mass, 15.8655, axis = 0)
    p84 = scoreatpercentile(MB_high_mass, 84.1345, axis = 0)

    for j in range(6):
        plt.plot([j + 0.1*i]*2, [p16[j], p84[j]], color = colors[i])

        plt.text(j, -19.2, ["CSP", "CfA1", "CfA2", "CfA3 KC", "CfA3 4S", "CfA4"][j])
yticks = plt.yticks()[0]

#plt.yticks(yticks, [""]*len(yticks))
plt.xticks([])

plt.legend(loc = 'best')
plt.savefig("compare.pdf")
plt.close()

