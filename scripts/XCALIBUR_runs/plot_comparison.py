import gzip
import pickle
from numpy import *
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
import tqdm


pfls = ["orig/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_v1_CfA4_v1.pickle",
        "Lin/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_v1_CfA4_v1.pickle",
        #"XCABNOSHIFTNOPOSMAGCUT/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_v1_CfA4_v1.pickle",
        "XCABNOSHIFTNOPOS/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_v1_CfA4_v1.pickle",
        "XCABNOSHIFT/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_v1_CfA4_v1.pickle",
        "XCAB/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_v1_CfA4_v1.pickle"]
        #"meanstarX/samples_CSP_v1_CfA1_v1_CfA2_v1_CfA3_4S_v1_CfA3_KC_v1_CfA4_v1_SDSS_v1.pickle"]
    

#pfls = ["meanstar/samples_CSP_v1_CfA1_v1_CfA1_STD_v1_CfA2_v1_CfA3_4S_v1_CfA3_KC_v1_CfA4_v1.pickle"]

colors = 'krmcgb'

n_samples = 5

labels = {"orig": "Smith/Landolt $\\rightarrow$ BD+17",
          "Lin": "Smith/Landolt $\\rightarrow$ Many Stars",
          "XCABNOSHIFTNOPOSMAGCUT": "X-CALIBUR w/o $\Delta \lambda$ w/o Position$\\rightarrow$ Six Stars, Mag Cut",
          "XCABNOSHIFTNOPOS": "X-CALIBUR w/o $\Delta \lambda$ w/o Position$\\rightarrow$ Six Stars",
          "XCABNOSHIFT": "X-CALIBUR w/o $\Delta \lambda$ $\\rightarrow$ Six Stars",
          "XCAB": "X-CALIBUR $\\rightarrow$ Six Stars"}

blinding_const = -19.15 + random.normal()*0.05


for i in tqdm.tqdm(range(len(pfls))):
    fit_params = pickle.load(gzip.open(pfls[i], 'rb'))
    

    MB_high_mass = fit_params["MB"] - blinding_const

    plt.plot(arange(5) + 0.1*i, median(MB_high_mass, axis = 0), 'os*.v^'[i], label = labels[pfls[i].split("/")[0]], color = colors[i])
    p16 = scoreatpercentile(MB_high_mass, 15.8655, axis = 0)
    p84 = scoreatpercentile(MB_high_mass, 84.1345, axis = 0)

    for j in range(len(p16)):
        plt.plot([j + 0.1*i]*2, [p16[j], p84[j]], color = colors[i])

        plt.text(j, -18.98 - blinding_const, ["CSP", "CfA1", "CfA2", "CfA3", "CfA4"][j])# 4S", "CfA3 KC", "CfA4"][j])
yticks = plt.yticks()[0]

#plt.yticks(yticks, [""]*len(yticks))
plt.xticks([])
plt.ylabel("Inferred Absolute Magnitude + Blinding Constant")

plt.legend(loc = 'best')
plt.savefig("compare.pdf", bbox_inches = 'tight')
plt.close()

