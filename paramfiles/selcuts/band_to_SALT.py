import subprocess
import sys
from FileRead import read_param
from astropy import cosmology
from numpy import *
import os

def do_it(cmd):
    print(cmd)
    print(subprocess.getoutput(cmd))


def get_SALT(X1, color):
    do_it("cd tmp\npython $PATHMODEL/python_code/SALT22wrapper.py -p DayMax 0 0.001 -p X1 " + str(X1) + " 0.001 -w 2000 9000 -f Color " + str(color))
    
    mB = read_param("tmp/result_salt2.dat", "RestFrameMag_0_B")
    return mB


def print_line(redshift, instr, band, magsys):

    do_it("rm -fr tmp")
    do_it("mkdir tmp")
    
    f = open("tmp/lc2fit_" + instr + "_" + band + ".dat", 'w')
    f.write("""#Date  :
#Mag  :
#Magerr  :
#end  :
@INSTRUMENT  """ + instr + """
@BAND  """ + band + """
@MAGSYS  """ + magsys + """
0  20  0.1
""")
    f.close()

    f = open("tmp/lightfile", 'w')
    f.write("Redshift  " + str(redshift) + '\n')
    f.write("MWEBV  0.0\n")
    f.close()
    
    mB0 = get_SALT(X1 = 0, color = 0)
    mB_c01 = get_SALT(X1 = 0, color = 0.1)

    print(mB0)
    print(mB_c01)

    cosmo = cosmology.FlatLambdaCDM(Om0 = 0.3, H0 = 70)
    mBcosmo = cosmo.distmod(z=redshift).value - 19.1

    print("z=%.3f %10s %10s %10s mobs = mB + %.3f + %.3f*c = %.3f + %.3f*c mBcosmo = %.2f" % (redshift, instr, band, magsys, 20 - mB0, (mB0 - mB_c01)/0.1, 20 - mB0 + mBcosmo, (mB0 - mB_c01)/0.1, mBcosmo))
    return redshift, instr, band, 20 - mB0, (mB0 - mB_c01)/0.1 # 20 - mB0 means (mobs - mB = this) or (mobs = mB + this)

for redshift, instr, band, magsys in [
        (0.04, "STANDARD", "R", "VEGA"), # Nearby
        (0.1, "STANDARD", "R", "VEGA"), # C/T
        (0.14, "STANDARD", "R", "VEGA"), # C/T
        (0.4, "SDSS", "SDSS_g", "AB"), # SDSS
        (0.5, "SDSS", "SDSS_r", "AB"), # SDSS
        (0.4, "SDSS", "SDSS_r", "AB"), # SDSS
        (0.01, "SDSS", "SDSS_r", "AB"), # SDSS
        (0.5, "STANDARD", "R", "VEGA"), # ESSENCE
        (0.6, "STANDARD", "R", "VEGA"), # ESSENCE
        (0.7, "STANDARD", "R", "VEGA"), # ESSENCE
        (0.8, "STANDARD", "R", "VEGA"), # ESSENCE
        (0.82, "STANDARD", "R", "VEGA"), # ESSENCE
        (0.7, "MEGACAMJLA", "i", "AB"), # SNLS 24.3 i-band
        (0.8, "MEGACAMJLA", "i", "AB"), # SNLS 24.3 i-band
        (1.0, "MEGACAMJLA", "i", "AB"), # SNLS 24.3 i-band
        (1.0, "STANDARD", "I", "VEGA"), # High-Redshift Ground
        (1.2, "STANDARD", "I", "VEGA"), # High-Redshift Ground
        (1.5, "ACSWF3pix", "F850LP", "VEGAHST"), # HST ACS
        (1.7, "ACSWF3pix", "F850LP", "VEGAHST") # HST ACS
        ]:
    

    print_line(redshift, instr, band, magsys)


to_run = [("HSC", "HSC_z", 1.8, "AB"),
          ("ACSWF", "F850LP", 1.8, "AB"),
          ("WFC3", "WFC3_f125w", 3.1, "AB"),
          ("SDSS", "SDSS_g", 1.0, "AB"),
          ("SDSS", "SDSS_r", 1.0, "AB"),
          ("SDSS", "SDSS_i", 1.0, "AB"),
          ("MEGACAMJLA", "i", 1.2, "AB"),
          ("CalanTololo", "CalanTololo_R", 1.0, "VEGA"),
          ("CalanTololo", "CalanTololo_I", 1.5, "VEGA")]



for item in to_run:
    [instr, band, zmax, magsys] = item
    the_name = (instr + "_")*(1 - band.count(instr)) + band
    
    f = open(the_name + "_selection.txt", 'w')
    f.write("#redshift  mobs = mB + col2 + col3*c\n")
    
    for redshift in arange(0.0001 + 0.2*(zmax > 1.7) + 0.3*(zmax > 2.), zmax + 0.1, 0.1):
        redshift, instr, band, magoff, cdep = print_line(redshift, instr, band, magsys)
        f.write("%f  %f  %f\n" % (redshift, magoff, cdep))
    f.close()

