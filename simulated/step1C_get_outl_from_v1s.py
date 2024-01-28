import glob
from FileRead import read_param, readcol, writecol
import tqdm

all_v1 = []
total = []
outliers = []

for sn_input in tqdm.tqdm(glob.glob("#v1 in tqdm.tqdm(glob.glob("*v1.txt")):
    [SNe] = readcol(v1, 'a')

    # $UNION/dataset_V_099/SN0123    
    # UNITY_V_099/SN_params/params_0104.dat

    outl = []
    for SN in SNe:
        pfl = SN.replace("$UNION/dataset", "UNITY").replace("/SN", "/SN_params/params_") + ".dat"

        outl.append(read_param(pfl, "outlier"))

    all_v1.append(v1)
    total.append(len(SNe))
    outliers.append(sum(outl))

writecol("outliers_by_dataset.txt", [all_v1, total, outliers])

    
