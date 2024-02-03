import glob
from FileRead import read_param, readcol, writecol
import tqdm
import numpy as np


unique_SNe = []


for sn_input in tqdm.tqdm(glob.glob("UNITY*/sn_input.txt")):
    [SNe, RA, DEC, ZHEL, ZCMB, passing] = readcol(sn_input, 'affffa')
    assert len(RA) > 0
    
    passing = np.array(passing)
    SNe = np.array(SNe)
    
    inds = np.where(passing == "True")

    unique_SNe.extend(SNe[inds])

all_v1 = {}
unique_SNe = np.unique(unique_SNe)

for SN in tqdm.tqdm(unique_SNe):
    # $UNION/dataset_V_099/SN0123    
    # UNITY_V_099/SN_params/params_0104.dat

    v1 = SN.split("/")[0] + "_v1.txt"

    pfl = SN.replace("dataset", "UNITY").replace("/SN", "/SN_params/params_") + ".dat"

    if v1 in all_v1:
        all_v1[v1][0] += 1
        all_v1[v1][1] += read_param(pfl, "outlier")
    else:
        all_v1[v1] = [1, read_param(pfl, "outlier")]
        
f = open("outliers_by_dataset.txt", 'w')
for key in all_v1:
    f.write(key + "  " + str(all_v1[key][0]) + "  " + str(all_v1[key][1]) + '\n')
f.close()
