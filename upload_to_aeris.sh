#!/usr/bin/env bash
# short shell script to manually upload data from eurec4a to aeris
# before uploading, check if the folder you want to upload already exists
# connect to aeris via lftp and cd to where the folder should be
# if not there mkdir "dirname" to create it
DAY=$(date +%Y%m%d)
YYYYMMDD=$(date +%Y%m%d --date="$DAY")

HOST="ftp.climserv.ipsl.polytechnique.fr"  # define host server -> aeris
RUSER="eurec4a"  # define user
PASS="pass4eurec4a!"  # define password
LCD="/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP"  # define local directory
RCD="/upload/SHIPS/RV-METEOR/DSHIP"  # define remote directory


lftp -c "open ftp://$RUSER:$PASS@$HOST;
lcd $LCD;
cd $RCD;
mkdir ${YYYYMMDD};
cd ${YYYYMMDD};
put ${new_3hr_name};
"
echo "Done uploading the newest hydrometeor fraction data and newest time height plots (3km and 15km)"