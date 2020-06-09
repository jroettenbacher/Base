# optimized for python3
# Friedhelm Jansen
#
"""
Plotting Ceilo NetCDF File
"""
# -*- coding: utf-8 -*-
from tkinter import *
import os, sys, time, select
import matplotlib
matplotlib.use('Agg')
import os.path ,glob, shutil
import tarfile, bz2
from scipy.io.netcdf import netcdf_file as Dataset
from netCDF4 import Dataset
from datetime import datetime
from numpy import *
from pylab import *
import tempfile
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.ticker import AutoMinorLocator,MultipleLocator
from matplotlib import gridspec
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
from numpy import loadtxt
from scipy.ndimage.filters import gaussian_filter
from matplotlib.pyplot import contour, show
from scipy.ndimage.measurements import mean, median
from scipy.interpolate import spline
from scipy.signal import savgol_filter
import math

####################     Parser       ##################################
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-t", "--time", dest="yyyymmdd", type=str,
     help="time of interest in the format yyyymmdd")

options = parser.parse_args()
########################################################################

yyyymmdd    = options.yyyymmdd
date_str    = yyyymmdd[2:8]
year_str    = yyyymmdd[0:4]
month_str   = yyyymmdd[4:6]
day_str     = yyyymmdd[6:8]

tmpExtract      = './'
outputFiles     = './'
Save_Images     = outputFiles

StartTime       = 0             #Alle Angaben in Stunden
EndTime         = 24
location        = 'Hamburg, Geomatikum'
instrument      = 'CHM170159'

# Read Data File
nc_file = ('/Users/friedhelmjansen/Desktop/BCO/Ceilo_ql/20'+date_str+'_Hamburg_CHM170159.nc')
print(nc_file)
with Dataset(nc_file, mode='r') as f:
    ceilo_time = f.variables['time'][:]
    range = f.variables['range'][:]
    back = f.variables['beta_raw'][:]
    chm_cbh = f.variables['cbh'][:]
    ceilo_secs  = mdate.epoch2num(ceilo_time)
    chm_cbh1 = np.array(chm_cbh[0])
    chm_cbh2 = np.array(chm_cbh[1])
#%%
#==========================================
# # Set starttime and endtime for one day:
#==========================================

StartTime=int((len(ceilo_time)/24*StartTime))
EndTime=int((len(ceilo_time)/24*EndTime))

#--------Plot Figure------------

print('Plot Figure')

# plot
#### LAYOUT
#Setting Fontsizes:
ax_title_size = 16
ylabel_size = 14
cb_title_size = 11
cb_size = 11
box_font = 11
legend_size = 11
# height for y-axis cbh-plot  
cbhlim = 4000
# height for backscatter-plot
betalim = 4200
#####


#Building the frame and setting Title
fig = plt.figure(1,figsize=[16,9],facecolor="#CCCCCC")
gs1 = gridspec.GridSpec(56, 100)
plt.suptitle('Ceilometer, '+location+', '+day_str+'.'+month_str+'.'+year_str, fontsize=30,y=0.99)
plt.subplots_adjust(top=0.97,bottom=0.02,left =0.05,right=0.97)

#patches=[]

ax1 = fig.add_subplot(gs1[5:40,5:-10]) #Backscatter
ax_cb = fig.add_subplot(gs1[5:40,80:92]) #Colorbar for Reflectivity
ax2 = fig.add_subplot(gs1[41:52,5:-10], sharex = ax1) #CloudBase
#ax3 = ax2.twinx()
#Adjusting appearance for all subplots:
axes = [ax1,ax2]
date_fmt = '%H:%M'
date_formatter = mdate.DateFormatter(date_fmt)

for ax in axes:
    plt.sca(ax)
    plt.xticks(rotation=70)

    ax.grid()

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(date_formatter)
#    ax.xaxis.set_minor_locator(AutoMinorLocator(6))

#    ax.tick_params('both', length=3, width=1, which='minor',labelsize=ylabel_size)
    ax.tick_params('both', length=4, width=2, which='major',labelsize=ylabel_size)
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
                                                
ax2.set_xlabel('UTC Time',fontsize=ylabel_size)

#Setting up the x_axes for all plots:
ax2.set_xticks(np.arange(min(ceilo_secs), max(ceilo_secs),0.083334/2)) #Abstand xticks = 2h, 1 Minute = 0.000694
ax2.set_xlim(min(ceilo_secs),max(ceilo_secs))

# scale for cb-axis
xlevels = np.arange(0.2e7,4e7,1e6)
#=========================
#Plotting Ceilo:
#========================= 

print('Plotting Backscatter')
Legend_handle = True
im1 = ax1.contourf(ceilo_secs[StartTime:EndTime], range[StartTime:EndTime], back[StartTime:EndTime].transpose(), levels = xlevels, cmap = plt.cm.viridis)
ax1.set_title('background substracted and normalised by laser shot number and stddev', fontsize=ax_title_size)
ax1.yaxis.set_ticks(np.arange(200,betalim,500))
ax1.set_ylim(200,betalim)
ax1.set_ylabel('Altitude [m]',fontsize=ylabel_size)
ax1.set_yticklabels(np.arange(0,betalim,500))
ax1.yaxis.get_major_ticks()[0].label1.set_visible(False)
#make x-axe-labels invisible:
for label in ax1.xaxis.get_ticklabels():
    label.set_visible(False)

#collorbar settings:
ax_cb.set_xticks([])
ax_cb.set_yticks([])
ax_cb.set_visible(False)
cb1 = plt.colorbar(im1, ax = ax_cb, shrink=1)
cb1.set_label('Backscatter signal [SNR]',fontsize=ylabel_size)
cb1.ax.tick_params(labelsize=cb_size)


print('Plotting cloud base height')
Legend_handle = True
ax2.plot(ceilo_secs[StartTime:EndTime], chm_cbh[StartTime:EndTime,0], color='b', lw=1, label='cloud base height 1')
ax2.plot(ceilo_secs[StartTime:EndTime], chm_cbh[StartTime:EndTime,1], color='r', lw=1, label='cloud base height 2')
ax2.set_ylabel('cloud base height [m]', fontsize=ylabel_size)
ax2.yaxis.set_ticks(np.arange(0,cbhlim,1000))
ax2.set_ylim(0,cbhlim)
ax2.legend(loc='upper left', fontsize = legend_size,frameon=False)
#ax2.yaxis.get_major_ticks()[1].label1.set_visible(False)
print("Save image")
plt.savefig(Save_Images+instrument+'_Hamburg_Quicklook_20'+date_str+'.png',dpi=120, facecolor='lightgrey')
print("Saved")
plt.close()
