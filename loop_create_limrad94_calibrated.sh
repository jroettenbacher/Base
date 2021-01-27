#!/usr/bin/env ksh

cd /projekt1/remsens/work/jroettenbacher/Base
PATH='/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/cloudnet_input_heave_cor_ca'
for date in {117..126}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	/home/jroettenbacher/.conda/envs/jr_base/bin/python3 /projekt1/remsens/work/jroettenbacher/Base/create_limrad94_calibrated_eurec4a.py date=20200${date} path=${PATH} 2>&1 | tee /projekt1/remsens/work/jroettenbacher/Base/log/200${date}create_limrad94_calibrated_eurec4a_jr.log
	echo "Done with 20200${date}"
done

echo "##################################################"
echo "#     STARTING DATE 20200128                 #"
echo "##################################################"
/home/jroettenbacher/.conda/envs/jr_base/bin/python3 /projekt1/remsens/work/jroettenbacher/Base/create_limrad94_calibrated_eurec4a.py date=20200128 path=${PATH} 2>&1 | tee /projekt1/remsens/work/jroettenbacher/Base/log/200128_create_limrad94_calibrated_eurec4a_jr.log
echo "Done with 20200128"

for date in {201..229}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	/home/jroettenbacher/.conda/envs/jr_base/bin/python3 /projekt1/remsens/work/jroettenbacher/Base/create_limrad94_calibrated_eurec4a.py date=20200${date} path=${PATH} 2>&1 | tee /projekt1/remsens/work/jroettenbacher/Base/log/200${date}_create_limrad94_calibrated_eurec4a_jr.log
	echo "Done with 20200${date}"
done
