cd orig
export UNION=~/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_BD17_
python ../../read_and_sample.py ../paramfile.txt 1 > log.txt &
sleep 5

cd ../Lin
export UNION=~/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_LIN_
python ../../read_and_sample.py ../paramfile.txt 1 > log.txt &
sleep 5

cd ../XCAB
export UNION=~/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_XCAB_
python ../../read_and_sample.py ../paramfile.txt 1 > log.txt &
sleep 5

cd ../XCABNOSHIFT
export UNION=~/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_XCABNOSHIFT_
python ../../read_and_sample.py ../paramfile.txt 1 > log.txt &
