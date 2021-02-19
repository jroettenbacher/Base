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
LCD="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/upload_to_aeris_v1.1"  # LIMRAD94 heave corrected, original res data
#LCD="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_CEILOMETER/upload_to_aeris"  # ceilometer data (nc files)
#LCD="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/HATPRO/upload_to_aeris"  # HATPRO data and plots
#LCD="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP/upload_to_aeris"  # DSHIP data
#define remote directory
#RCD="/upload/SHIPS/RV-METEOR/cloudradar/ncfiles"  # cloud radar
RCD="/upload/SHIPS/RV-METEOR/cloudradar/heave_corr_high_res_ncfiles"  # cloud radar
#RCD="/upload/SHIPS/RV-METEOR/ceilometer/ncfiles"  # ceilometer nc files
#RCD="/upload/SHIPS/RV-METEOR/radiometer"  # HATPRO data and plots, level 1 + 2
#RCD="/upload/SHIPS/RV-METEOR/DSHIP"  # DSHIP data
lftp -e "mirror -R $LCD $RCD " -u "$RUSER","$PASS" "$HOST"
