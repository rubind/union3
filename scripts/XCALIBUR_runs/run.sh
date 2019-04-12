cd orig
export UNION=/Users/rubind/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_BD17_
python ../../read_and_sample.py ../paramfile.txt 1 > log.txt

cd ../Lin
export UNION=/Users/rubind/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_LIN_
python ../../read_and_sample.py ../paramfile.txt 1 > log.txt

cd ../XCAB
export UNION=/Users/rubind/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_XCAB_
python ../../read_and_sample.py ../paramfile.txt 1 > log.txt

