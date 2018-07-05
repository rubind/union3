cd orig
export UNION=/Users/rubind/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_BD17_
python ../read_and_sample.py ../paramfile.txt 1 > log.txt

cd ../meanstar
export UNION=/Users/rubind/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_LINMEANSTAR_
python ../read_and_sample.py ../paramfile.txt 1 > log.txt

cd ../meanstarX
export UNION=/Users/rubind/Dropbox/Shared/Union3_Photometry/Union3_XCALIBUR_MEANSTAR_
python ../read_and_sample.py ../paramfile.txt 1 > log.txt

