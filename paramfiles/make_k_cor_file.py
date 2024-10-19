import sncosmo
import Spectra
import numpy as np
import os
import sys


def get_B0(x0_mag, c, z):
    model.set(x0 = 10.**(-0.4*x0_mag), x1 = 0.0, c = c, t0 = 0.0, z = 0.0)

    mags = {}
    for filt in ["ux", "b", "v", "r"]:
        mags[filt] = model.bandmag('bessell' + filt, 'vega', 0.0) + 0.03

    model.set(x0 = 10.**(-0.4*x0_mag), x1 = 0.0, c = c, t0 = 0.0, z = z)
    mags[band] = model.bandmag(band, magsys, 0.0)
    
    return mags


salt2_version = "salt3-f22"
band = sys.argv[1]
magsys = sys.argv[2]
zmax = float(sys.argv[3])
source = sncosmo.SALT3Source(modeldir = os.environ["PATHMODEL"] + "/" + salt2_version + "/")
model = sncosmo.Model(source)

zs = []
col2 = []
col3 = []


for z in np.arange(0.0, zmax + 0.001, 0.01):
    try:
        mags_c0 = get_B0(x0_mag = 5., c = 0., z = z)
        mags_c01 = get_B0(x0_mag = 5., c = 0.1, z = z)

        zs.append(z)
        col2.append(mags_c0[band] - mags_c0["b"])
        col3.append((mags_c01[band] - mags_c0[band])/0.1)
    except:
        print("Skipping ", z)
    
f = open(band + "_selection_%s_%s.txt" % (salt2_version, magsys), 'w')
f.write("#redshift  mobs = mB + col2 + col3*c\n")
f.write("%.6f  %.6f  %.6f\n" % (-0.01, col2[0], col3[0]))

for i in range(len(zs)):
    f.write("%.6f  %.6f  %.6f\n" % (zs[i], col2[i], col3[i]))
f.close()
