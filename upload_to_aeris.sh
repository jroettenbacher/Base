#!/usr/bin/env bash
# short shell script to upload data from eurec4a to aeris by mirroring a whole folder
# before uploading, check if the folder you want to upload to already exists
# connect to aeris via lftp and cd to where the folder should be
# if not there mkdir "dirname" to create it
# or use WinSCP for that

DAY=$(date +%Y%m%d)
YYYYMMDD=$(date +%Y%m%d --date="$DAY")

HOST="ftp.climserv.ipsl.polytechnique.fr"  # define host server -> aeris
RUSER="eurec4a"  # define user
PASS="pass4eurec4a!"  # define password
LCD="/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP/upload_to_aeris"  # define local directory
RCD="/upload/SHIPS/RV-METEOR/DSHIP"  # define remote directory

lftp -e "mirror -R $LCD $RCD " -u "$RUSER","$PASS" "$HOST"
