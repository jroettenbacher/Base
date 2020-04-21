#!/bin/bash
# loop through days and run LIMRAD94_to_Cloudnet_v2.py

cd /projekt1/remsens/work/jroettenbacher/Base
for date in {117..126}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	/home/jroettenbacher/envs/jr_base/bin/python3 /projekt1/remsens/work/jroettenbacher/Base/LIMRAD94_to_Cloudnet_v2.py date=20200${date} > /projekt1/remsens/work/jroettenbacher/Base/log/limrad94_to_cloudnet_200${date}.log 2>&1
	echo "Done with 20200${date}"
done
echo "##################################################"
	echo "#     STARTING DATE 20200128                 #"
	echo "##################################################"
	/home/jroettenbacher/envs/jr_base/bin/python3 /projekt1/remsens/work/jroettenbacher/Base/LIMRAD94_to_Cloudnet_v2.py date=20200128 > /projekt1/remsens/work/jroettenbacher/Base/log/limrad94_to_cloudnet_200128.log 2>&1
	echo "Done with 20200128"
for date in {201..229}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	/home/jroettenbacher/envs/jr_base/bin/python3 /projekt1/remsens/work/jroettenbacher/Base/LIMRAD94_to_Cloudnet_v2.py date=20200${date} > /projekt1/remsens/work/jroettenbacher/Base/log/limrad94_to_cloudnet_200${date}.log 2>&1
	echo "Done with 20200${date}"
done
