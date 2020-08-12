#!/usr/bin/env ksh

cd /projekt1/remsens/work/jroettenbacher/Base

for date in {201..229}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	/home/jroettenbacher/envs/jr_base/bin/python3 /projekt1/remsens/work/jroettenbacher/Base/create_limrad94_calibrated.py date=20200${date} > /projekt1/remsens/work/jroettenbacher/Base/log/create_limrad94_calibrated_200${date}.log 2>&1
	echo "Done with 20200${date}"
done