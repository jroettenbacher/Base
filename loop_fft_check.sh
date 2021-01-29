#!/usr/bin/env bash

today=$(date +%Y%m%d)
version=("jr" "ca")
home=/projekt1/remsens/work/jroettenbacher/Base
cd $home
for v in ${version[*]}; do
  for date in {117..126}; do
    echo "##################################################"
    echo "#     STARTING DATE 20200${date}                 #"
    echo "##################################################"
    /home/jroettenbacher/.conda/envs/jr_base/bin/python3 ${home}/heave_cor_ff_check.py date=20200${date} version=${v} 2>&1 | tee ${home}/log/${today}heave_cor_ff_check_200${date}.log
    echo "Done with 20200${date}"
  done

  for date in {201..229}; do
    echo "##################################################"
    echo "#     STARTING DATE 20200${date}                 #"
    echo "##################################################"
    /home/jroettenbacher/.conda/envs/jr_base/bin/python3 ${home}/heave_cor_ff_check.py date=20200${date} version=${v} 2>&1 | tee ${home}/log/${today}heave_cor_ff_check_200${date}.log
    echo "Done with 20200${date}"
  done
  echo "##################################################"
  echo "#     DONE WITH VERSION ${v}                 #"
  echo "##################################################"
done