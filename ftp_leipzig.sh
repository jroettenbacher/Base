#!/bin/bash 
# Parse command-line arguments
if [ "$1" = "--date" ]; then
	DAY=$2
	echo "Uploading quicklooks from ${DAY} for LIMHAT and LIMRAD94 to ftp.uni-leipzig.de..."
else
	DAY=$(date +%Y%m%d --date='yesterday')
	echo "Uploading yesterdays quicklooks for LIMHAT and LIMRAD94 to ftp.uni-leipzig.de..."
fi

YYMM=$(date +%y%m --date=$DAY)
YYMMDD=$(date +%y%m%d --date=$DAY)
HOST="ftp.uni-leipzig.de"
RUSER="meteo"
PASS="W1lia24;"
LCD1="/home/remsens/code/larda3/scripts/plots"
LCD2="/home/remsens/data/LIMHAT/met/plots/level2/$YYMM"
RCD1="/kalesse/EUREC4A_data/LIMRAD94/plots"
RCD2="/kalesse/EUREC4A_data/LIMHAT/plots/level2/$YYMM"
# getting new LIMRAD94 plots and renaming them
new_3hr_plot=$(ls -t $LCD1/radar_hydro_frac/$DAY*_3hr.png | head -n1)
new_1hr_plot=$(ls -t $LCD1/radar_hydro_frac/$DAY*_1hr.png | head -n1)
new_3hr_file=$(ls -t $LCD1/radar_hydro_frac/$DAY*_3hr.txt | head -n1)
new_15km_Z_plot=$(ls -t $LCD1/$DAY*_15km_Z.png | head -n1)
new_3km_Z_plot=$(ls -t $LCD1/$DAY*_3km_Z.png | head -n1)

# getting new LIMHAT plots
# save all seven plots in list to loop over them
FILES=$(ls -t $LCD2/${YYMMDD}_met*.png | head -n7)

# upload LIMRAD94 plots and file
echo "Uploading ${new_3hr_plot}, ${new_1hr_plot}, ${new_3hr_file}, ${new_15km_Z_plot} and ${new_3km_Z_plot} to $RCD."
# connect to ftp server, change into directories and upload
lftp -c "open ftp://$RUSER:'$PASS'@$HOST; 
lcd $LCD1/upload;
cd $RCD1;
put ${new_15km_Z_plot};
echo 'Done putting ${new_15km_Z_plot}';
put ${new_3km_Z_plot};
echo 'Done putting ${new_3km_Z_plot}'
lcd $LCD1/radar_hydro_frac
cd $RCD1/radar_hydro_frac
put ${new_3hr_plot};
echo 'Done putting ${new_3hr_plot}';
put ${new_1hr_plot};
echo 'Done putting ${new_1hr_plot}';
put ${new_3hr_file};
echo 'Done putting ${new_3hr_file}';
"
echo "Done uploading the newest hydrometeor fraction data and newest time height plots (3km and 15km)"

# upload LIMHAT plots
echo "Uploading new LIMHAT plots to $RCD2..."
for FILE in ${FILES}; do
	lftp -c "open ftp://$RUSER:'$PASS'@$HOST;
	lcd $LCD2;
	cd $RCD2;
	put ${FILE};
	echo 'Done putting ${FILE}';
	"
done  #FILES

echo "Done uploading all new LIMRAD94 and LIMHAT files. See you again soon!"