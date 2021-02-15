#!/usr/bin/env ksh

home=/projekt1/remsens/work/jroettenbacher/Base
python=/home/jroettenbacher/.conda/envs/jr_base/bin/python3
cd ${home}
version="ca"
path="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/upload_to_aeris_v1"
for date in {117..126}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	${python} ${home}/create_limrad94_calibrated_eurec4a.py date=20200${date} path=${path} heave_corr_version=${version} 2>&1 | tee ${home}/log/200${date}create_limrad94_calibrated_eurec4a_${version}.log
	echo "Done with 20200${date}"
done

echo "##################################################"
echo "#     STARTING DATE 20200128                 #"
echo "##################################################"
${python} ${home}/create_limrad94_calibrated_eurec4a.py date=20200128 path=${path} heave_corr_version=${version} 2>&1 | tee ${home}/log/200128_create_limrad94_calibrated_eurec4a_${version}.log
echo "Done with 20200128"

for date in {201..229}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	${python} ${home}/create_limrad94_calibrated_eurec4a.py date=20200${date} path=${path} heave_corr_version=${version} 2>&1 | tee ${home}/log/200${date}_create_limrad94_calibrated_eurec4a_${version}.log
	echo "Done with 20200${date}"
done
