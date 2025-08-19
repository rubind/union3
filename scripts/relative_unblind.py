from subprocess import getoutput
import numpy as np
import sys

print("python relative_unblind.py Om 1.05 */log.txt")

par_name = sys.argv[1]
max_Rhat = float(sys.argv[2])
logs = sys.argv[3:]

grepout = [getoutput("grep '%s ' %s" % (par_name, item)) for item in logs]

grepout = [item for item in grepout if item != ""]

print("len", len(grepout))

for i in range(len(grepout))[::-1]:
    parsed = grepout[i].split(None)
    Rhat = float(parsed[-1])
    if Rhat > max_Rhat:
        del grepout[i]
        print("bad Rhat")
print("len", len(grepout))


vals = [float(item.split(None)[1]) for item in grepout]
uncs = [float(item.split(None)[3]) for item in grepout]

print("Max - Min ", par_name, np.max(vals) - np.min(vals))
print("Mean/median uncertainty ", np.mean(uncs), np.median(uncs))
