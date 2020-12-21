#!/usr/bin/env bash

today=$(date +%Y%m%d)
home=/projekt1/remsens/work/jroettenbacher/Base
cd $home
for date in {117..126}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	/home/jroettenbacher/.conda/envs/jr_base/bin/python3 ${home}/make_virga_mask.py date=20200${date} 2>&1 | tee ${home}/log/${today}make_virga_mask_200${date}.log
	echo "Done with 20200${date}"
done

for date in {201..229}; do
	echo "##################################################"
	echo "#     STARTING DATE 20200${date}                 #"
	echo "##################################################"
	/home/jroettenbacher/.conda/envs/jr_base/bin/python3 ${home}/make_virga_mask.py date=20200${date} 2>&1 | tee ${home}/log/${today}make_virga_mask_200${date}.log
	echo "Done with 20200${date}"
done