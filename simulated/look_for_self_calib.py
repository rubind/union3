from subprocess import getoutput
import sys
import numpy as np

all_calibs = {}


for logfl in sys.argv[1:]:
    calibs = getoutput("grep calibs " + logfl).split('\n')
    calib_names = getoutput("grep calib_names " + logfl)
    calib_names = eval(calib_names.split('the_data["calib_names"]')[1].strip())

    print("calib_names", calib_names)
    assert len(calib_names) == len(calibs)

    for i, key in enumerate(calib_names):
        if key in all_calibs:
            all_calibs[key].append(float(calibs[i].split(None)[1]))
        else:
            all_calibs[key] = [float(calibs[i].split(None)[1])]

print("all_calibs", all_calibs)

all_rms = []
for key in all_calibs:
    all_rms.append([np.std(all_calibs[key], ddof=1), key])

all_rms.sort()

for item in all_rms:
    print(item)
    
