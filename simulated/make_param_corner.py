import sys
import numpy as np
import subprocess

params = ["alpha_fast", "alpha_slow", "beta_B", "beta_R_high", "beta_R_low", "x1_star_fast", "x1_star_slow", "R_x1_fast", "R_x1_slow", "c_star_fast", "c_star_slow", "R_c_fast", "R_c_slow", "MB_fast_minus_slow", "Om", "H0", "sigma_int_fast", "outl_frac", "delta_0", "delta_h", "mBx1c_int_variance\[1\]", "mBx1c_int_variance\[2\]", "mBx1c_int_variance\[3\]"]

vals = [[] for param in params]

for dr in sys.argv[1:]:
    for i in range(len(params)):
        val = subprocess.getoutput("grep '" + params[i] + " ' " + dr + "/log.txt").split(None)
        if len(val) > 5:
            vals[i].append(float(val[1]))

            assert val[0] == params[i].replace("\\", ""), str(vals)


vals = np.array(vals)


print(vals)

cor = np.corrcoef(vals)

print(vals.shape)
print(cor.shape)
print(len(params))

cor_param_param = []
for i in range(len(params)):
    for j in range(i+1, len(params)):
        cor_param_param.append((np.abs(cor[i,j]), cor[i,j], params[i], params[j]))

cor_param_param.sort()

for item in cor_param_param[::-1]:
    print(" "*(item[1] > 0) + "%.3f %s %s" % item[1:])
    
