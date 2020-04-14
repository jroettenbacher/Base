#!/usr/bin/env bash
# short shell script to upload data from eurec4a to aeris by mirroring a whole folder
# before uploading, check if the folder you want to upload to already exists
# connect to aeris via lftp and cd to where the folder should be
# lftp eurec4a@ftp.climserv.ipsl.polytechnique.fr
# password: pass4eurec4a!
# if not there, mkdir "dirname" to create it
# or use WinSCP for that

DAY=$(date +%Y%m%d)
YYYYMMDD=$(date +%Y%m%d --date="$DAY")

HOST="ftp.climserv.ipsl.polytechnique.fr"  # define host server -> aeris
RUSER="eurec4a"  # define user
PASS="pass4eurec4a!"  # define password
# define local directory
#LCD="/projekt2/remsens/data/campaigns/eurec4a/LIMRAD94/upload_to_aeris"  # LIMRAD94 data (30s averages)
LCD="/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_CEILOMETER/upload_to_aeris"  # ceilometer data (nc files)
#define remote directory
#RCD="/upload/SHIPS/RV-METEOR/cloudradar/ncfiles"  # cloud radar
RCD="/upload/SHIPS/RV-METEOR/ceilometer/ncfiles"  # ceilometer nc files
lftp -e "mirror -R $LCD $RCD " -u "$RUSER","$PASS" "$HOST"
