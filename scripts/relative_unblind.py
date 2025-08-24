from subprocess import getoutput
import numpy as np
import sys
import tqdm

print("python relative_unblind.py Om 1.05 */log.txt")

par_name = sys.argv[1]
max_Rhat = float(sys.argv[2])
logs = sys.argv[3:]

grepout = []

for item in tqdm.tqdm(logs):
    grepout.append(getoutput("grep '%s ' %s" % (par_name, item)))

for i in range(len(grepout))[::-1]:
    if grepout[i] == "" or grepout[i].count("directory"):
        del grepout[i]
        del logs[i]

print("len", len(grepout))

for i in range(len(grepout))[::-1]:
    parsed = grepout[i].split(None)
    Rhat = float(parsed[-1])
    if Rhat > max_Rhat:
        del grepout[i]
        del logs[i]
        print("bad Rhat")
print("len", len(grepout))
assert len(logs) == len(grepout)

prefixes = [item.split("/")[0][:-4] for item in logs]
for item in set(prefixes):
    print(item, prefixes.count(item))

vals = [float(item.split(None)[1]) for item in grepout]
uncs = [float(item.split(None)[3]) for item in grepout]

assert len(prefixes) == len(vals)

average_vals = []
for item in set(prefixes):
    vals_for_prefix = []
    print(item, prefixes.count(item))
    if prefixes.count(item) > 0:
        for i in range(len(prefixes)):
            if prefixes[i] == item:
                vals_for_prefix.append(vals[i])
    average_vals.append(np.mean(vals_for_prefix))
    
print("Max - Min ", par_name, np.max(average_vals) - np.min(average_vals))
print("Mean/median uncertainty ", np.mean(uncs), np.median(uncs))
