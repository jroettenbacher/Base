#!/bin/bash 
cd /home/remsens/code/larda3/scripts
#DAY=$(date +%Y%m%d)
DAY=$1
YYYYMMDD=$(date +%Y%m%d --date=${DAY})
# /home/remsens/anaconda3/bin/python LIMRAD94_to_Cloudnet_v2.py date=$DAY > ~/log/limrad94_to_cloudnet.log 2>&1
/home/remsens/anaconda3/bin/python quick_cloud_stat.py $DAY 3
/home/remsens/anaconda3/bin/python quick_cloud_stat.py $DAY 1   
HOST="ftp.climserv.ipsl.polytechnique.fr"
RUSER="eurec4a"
PASS="pass4eurec4a!"
LCD="/home/remsens/code/larda3/scripts/plots"
RCD="/upload/SHIPS/RV-METEOR/cloudradar"
new_3hr_plot=$(ls -t $LCD/radar_hydro_frac/$DAY*_3hr.png | head -n1)
new_1hr_plot=$(ls -t $LCD/radar_hydro_frac/$DAY*_1hr.png | head -n1)
new_3hr_file=$(ls -t $LCD/radar_hydro_frac/$DAY*high_3hr.txt | head -n1)
new_1hr_file=$(ls -t $LCD/radar_hydro_frac/$DAY*high_1hr.txt | head -n1)
new_3hr_file_all=$(ls -t $LCD/radar_hydro_frac/$DAY*all_3hr.txt | head -n1)
new_1hr_file_all=$(ls -t $LCD/radar_hydro_frac/$DAY*all_1hr.txt | head -n1)
new_15km_Z_plot=$(ls -t $LCD/$DAY*_15km_Z.png | head -n1)
new_3km_Z_plot=$(ls -t $LCD/$DAY*_3km_Z.png | head -n1)
new_3hr_name=$(date +RV-Meteor_cloudradar_hydro-fraction_3hr_%Y%m%d.png --date=$DAY)
new_3hr_file_name=$(date +RV-Meteor_cloudradar_hydro-fraction_3hr_%Y%m%d.txt --date=$DAY)
new_1hr_name=$(date +RV-Meteor_cloudradar_hydro-fraction_1hr_%Y%m%d.png --date=$DAY)
new_1hr_file_name=$(date +RV-Meteor_cloudradar_hydro-fraction_1hr_%Y%m%d.txt --date=$DAY)
new_3hr_file_all_name=$(date +RV-Meteor_cloudradar_hydro-fraction_all_3hr_%Y%m%d.txt --date=$DAY)
new_1hr_file_all_name=$(date +RV-Meteor_cloudradar_hydro-fraction_all_1hr_%Y%m%d.txt --date=$DAY)
new_15km_Z_name=$(date +RV-Meteor_cloudradar_Ze_15km_%Y%m%d.png --date=$DAY)
new_3km_Z_name=$(date +RV-Meteor_cloudradar_Ze_3km_%Y%m%d.png --date=$DAY)
echo "Copying to ${LCD}/upload and renaming..."
cp ${new_3hr_plot} $LCD/upload/${new_3hr_name}
cp ${new_3hr_file} $LCD/upload/${new_3hr_file_name}
cp ${new_1hr_plot} $LCD/upload/${new_1hr_name}
cp ${new_1hr_file} $LCD/upload/${new_1hr_file_name}
cp ${new_15km_Z_plot} $LCD/upload/${new_15km_Z_name}
cp ${new_3km_Z_plot} $LCD/upload/${new_3km_Z_name}
cp ${new_3hr_file_all} $LCD/upload/${new_3hr_file_all_name}
cp ${new_1hr_file_all} $LCD/upload/${new_1hr_file_all_name}
echo "Uploading ${new_3hr_name}, ${new_1hr_name}, ${new_3hr_file_name}, ${new_3hr_file_all_name}, ${new_1hr_file_all_name}, ${new_15km_Z_name} and ${new_3km_Z_name} to $RCD."

lftp -c "open ftp://$RUSER:$PASS@$HOST; 
lcd $LCD/upload;
cd $RCD;
mkdir ${YYYYMMDD};
cd ${YYYYMMDD};
put ${new_3hr_name};
echo 'Done putting ${new_3hr_name}';
put ${new_1hr_name};
echo 'Done putting ${new_1hr_name}';
put ${new_3hr_file_name};
echo 'Done putting ${new_3hr_file_name}';
put ${new_1hr_file_name};
echo 'Done putting ${new_1hr_file_name}';
put ${new_3hr_file_all_name};
echo 'Done putting ${new_3hr_file_all_name}';
put ${new_1hr_file_all_name};
echo 'Done putting ${new_1hr_file_all_name}';

put ${new_15km_Z_name};
echo 'Done putting ${new_15km_Z_name}';
put ${new_3km_Z_name};
echo 'Done putting ${new_3km_Z_name}';
"
echo "Done uploading the newest hydrometeor fraction data and newest time height plots (3km and 15km)"
