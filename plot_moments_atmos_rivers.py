#!/usr/bin/python3

import sys

# just needed to find pyLARDA from this location
sys.path.append('../larda/')
sys.path.append('.')

import matplotlib

matplotlib.use('TkAgg')  # matplotlib.use('Agg') ' depends on Mac/Wind/Linux
import pyLARDA
import pyLARDA.helpers as h
import datetime
import scipy.ndimage as spn

import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

log = logging.getLogger('pyLARDA')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA('remote', uri='http://larda.tropos.de/larda3').connect('lacros_dacapo', build_lists=False)

# define start and end date
year = 2018
month = 12
day0 = 1
HH0 = 0
MM0 = 00
day1 = 2
HH1 = 23
MM1 = 59

dt_begin = datetime.datetime(year, month, day0, HH0, MM0, 0)
dt_end = datetime.datetime(year, month, day1, HH1, MM1, 0)
plot_range = [0, 12000]

# define plot name
name = f'plots/{dt_begin:%Y%m%d_%H%M}_{dt_end:%Y%m%d_%H%M}_'

#  read in moments: look into params_dacapo.toml (in larda-cfg folder)
system_Cloudnet = "CLOUDNET_LIMRAD"
MWR_LWP_DeBilt = larda.read(system_Cloudnet, "LWP", [dt_begin, dt_end], plot_range)
radar_Z = larda.read(system_Cloudnet, "Z", [dt_begin, dt_end], plot_range)
radar_Z['colormap'] = 'jet'

system_MIRA_Cloudnet = "CLOUDNET"
MIRA_Z = larda.read(system_MIRA_Cloudnet, "Z", [dt_begin, dt_end], plot_range)
MIRA_Z['colormap'] = 'jet'

system_Limrad94 = "LIMRAD94"
Radar_LWP_PuntaRetr = larda.read(system_Limrad94, "LWP", [dt_begin, dt_end], plot_range)
Radar_RR = larda.read(system_Limrad94, "rr", [dt_begin, dt_end], plot_range)
Radar_SurfT = larda.read(system_Limrad94, "SurfTemp", [dt_begin, dt_end], plot_range)
Radar_SurfRH = larda.read(system_Limrad94, "SurfRelHum", [dt_begin, dt_end], plot_range)
Radar_SurfWS = larda.read(system_Limrad94, "SurfWS", [dt_begin, dt_end], plot_range)
Radar_SurfT['var'] = Radar_SurfT['var'] - 273.15  # conversion to °C
T_max = (Radar_SurfT['var'].max()) + 3
RH_max = (Radar_SurfRH['var'].max()) + 3

system_MWR = "HATPRO"  # should be the values processed with Punta Arenas retrieval
MWR_IWV = larda.read(system_MWR, "IWV", [dt_begin, dt_end], plot_range)  # kg/m2
MWR_LWP = larda.read(system_MWR, "LWP", [dt_begin, dt_end], plot_range)  # kg/m2
MWR_flag = larda.read(system_MWR, "flag", [dt_begin, dt_end], plot_range)
MWR_LWP['var'] = MWR_LWP['var'] * 1000  # conversion to g/m2
MWR_LWP['var_unit'] = 'g m^-2'
iwv_max = (MWR_IWV['var'].max()) + 3
# mean_IWV_all = np.mean(MWR_IWV['var'])
# std_IWV_all  = np.var(MWR_IWV['var'])
# print(MWR_flag.shape);

# Set HATPRO internal rain flag (as done by Patric)
# The 4th bit of the flag variable lwp_flag must be set in case of rain
lwp_israin = [h.isKthBitSet(MWR_flag['var'][i], 4) for i in range(MWR_flag['ts'].size)]

# just copy the data into new variables for valid (no rain) and invalid (rain) periods
lwp_data_valid = MWR_LWP['var'] * 1.
lwp_data_invalid = MWR_LWP['var'] * 1.

lwp_data_valid[lwp_israin == 1] = np.nan
lwp_data_invalid[lwp_israin == 0] = np.nan

# create new larda container and put valid/invalid data to 'var'
lwp_container_valid = h.put_in_container(lwp_data_valid, MWR_LWP)
lwp_container_invalid = h.put_in_container(lwp_data_invalid, MWR_LWP)

system_disdro = "PARSIVEL"
disdro_RR = larda.read(system_disdro, "rainrate", [dt_begin, dt_end], plot_range)
disdro_RR['var'] = disdro_RR['var'] * 3.6e6  # conversion from m/s in mm/h: * 10^6 *1/3600
disdro_RainFlag = (disdro_RR['var'] > 0) * iwv_max
disdro_RainFlagMath = (disdro_RR['var'] > 0)  # --> True means > 0; False means <= 0

# system_SHAUN       = "SHAUN"
# SHAUN_advVel       = larda.read(system_disdro, "advection_vel", [dt_begin, dt_end], plot_range)

# convert unix time to datetime (for the list of 'ts')
dt_Cloudnet = [h.ts_to_dt(ts) for ts in MWR_LWP_DeBilt['ts']]
dt_MIRA_Cloudnet = [h.ts_to_dt(ts) for ts in MIRA_Z['ts']]
dt_Limrad94 = [h.ts_to_dt(ts) for ts in Radar_LWP_PuntaRetr['ts']]
dt_disdro = [h.ts_to_dt(ts) for ts in disdro_RR['ts']]
dt_MWR = [h.ts_to_dt(ts) for ts in MWR_IWV['ts']]
# dt_SHAUN    = [h.ts_to_dt(ts) for ts in SHAUN_advVel['ts']]

# set IWV and LWP values for disdrometer rainy pxl to NaN
# a) interpolate disdro time resolution to MWR time resolution + set MWR-IWV and MWR-LWP at disdro rainy profiles to NaN
f = interpolate.interp1d(disdro_RR['ts'], disdro_RR['var'], kind='nearest')
disdro_RR_ip2mwr = f(MWR_IWV['ts'])  # interpolate Disdro RR to MWR time values
# print(disdro_RR_ip2mwr.shape); print(MWR_IWV['ts'].shape); # --> same length of vectors checked: ok
disdro_RainFlag_ip2mwr = (disdro_RR_ip2mwr > 0)  # Rain Flag at MWR time steps
MWR_IWV['var'][disdro_RainFlag_ip2mwr] = np.nan  # MWR IWV at rainy profiles is set to NaN
MWR_LWP['var'][disdro_RainFlag_ip2mwr] = np.nan  # MWR LWP at rainy profiles is set to NaN
# b) interpolate disdro time resolution to LIMRAD94 time resolution and set 89GHz LWP at rainy profiles to NaN
f = interpolate.interp1d(disdro_RR['ts'], disdro_RR['var'], kind='nearest', bounds_error=False, fill_value=-1)
disdro_RR_ip2Limrad94 = f(Radar_LWP_PuntaRetr['ts'])  # interpolate Disdro RR to Limrad94 time steps
disdro_RainFlag_ip2Limrad94 = (disdro_RR_ip2Limrad94 > 0)  # Rain Flag at Limrad94 time steps
Radar_LWP_PuntaRetr['var'][disdro_RainFlag_ip2Limrad94] = np.nan  # 89GHz LWP at rainy profiles is set to NaN

# determine some simple statistics
mean_MWR_IWV = np.nanmean(MWR_IWV['var'])
print(mean_MWR_IWV)

std_MWR_IWV = np.nanstd(MWR_IWV['var'])
print(std_MWR_IWV)

mean_MWR_LWP = np.nanmean(MWR_LWP['var'])
print(mean_MWR_LWP)

std_MWR_LWP = np.nanstd(MWR_LWP['var'])
print(std_MWR_LWP)

'''# pick subset of time series for statistics: easier way: only choose ts I want statistics for in read in of time series
#dt_begin_stat   = h.dt_to_ts(datetime.datetime(year, month, 6, HH0, MM0, 0))
#dt_end_stat     = h.dt_to_ts(datetime.datetime(year, month, 6, HH1, MM1, 0))

dt_begin_idx = h.argnearest(MWR_IWV['ts'], dt_begin_stat)
dt_end_idx   = h.argnearest(MWR_IWV['ts'], dt_end_stat)

disdro_RainFlagMath_cut = disdro_RainFlagMath[dt_begin_idx:dt_end_idx]
MWR_IWV_cut             = MWR_IWV['var'][dt_begin_idx:dt_end_idx]
mean_IWV = np.mean(MWR_IWV_cut [disdro_RainFlagMath_cut])
std_IWV  = np.var(MWR_IWV_cut[disdro_RainFlagMath_cut])
print(mean_IWV)
print(std_IWV)'''

####################################################################################
# plotting section
####################################################################################

# 1.a) plot and save Limrad94 adar reflectivity figure
fig, ax = pyLARDA.Transformations.plot_timeheight(radar_Z, rg_converter=True,
                                                  title=True, z_converter='lin2z')
fig.savefig(name + system_Cloudnet + '_Z.png', dpi=250)

# 1.b) plot and save MIRA radar reflectivity figure
fig, ax = pyLARDA.Transformations.plot_timeheight(MIRA_Z, rg_converter=True,
                                                  title=True, z_converter='lin2z')
fig.savefig(name + system_MIRA_Cloudnet + '_MIRA_Z.png', dpi=250)

# 2.a) plot and save MWR_LWP_DeBilt figure
fig, ax = pyLARDA.Transformations.plot_timeseries(MWR_LWP_DeBilt, rg_converter=True,
                                                  title=True, zlim=[-3, 3])
ax.set_ylim([0, 2200])
fig.savefig(name + system_Cloudnet + '_MWR_LWP_DeBilt.png', dpi=250)

# 2.b) plot and save 89 GHz Radar_LWP_PuntaArenasRetrieval figure
fig, ax = pyLARDA.Transformations.plot_timeseries(Radar_LWP_PuntaRetr, rg_converter=True,
                                                  title=True, zlim=[-3, 3])
ax.set_ylim([0, 2200])
fig.savefig(name + system_Limrad94 + '_Radar_LWP_PuntaRetr.png', dpi=250)

# 2.c) v1: plot and save figure of LWP from MWR-DeBiltRetrieval, MWR Punta Retrieval AND 89 GHz Radar_LWP_PuntaArenasRetrieval
fig, ax = pyLARDA.Transformations.plot_timeseries(MWR_LWP_DeBilt, rg_converter=True,
                                                  title=True, zlim=[-3, 3])
ax.plot(dt_Limrad94, Radar_LWP_PuntaRetr['var'], color='black', linewidth=1.0)
ax.plot(dt_MWR, MWR_LWP['var'], color='red', linewidth=0.5)
ax.set_ylim([0, 2200])  # limit for left y-axis
ax.legend(['MWR DeBilt Retrieval', '89 GHz radar passive Ch., Punta Arenas Retrieval', 'MWR Punta Arenas Retrieval'])
fig.savefig(name + 'LWP_comparison.png', dpi=250)

# 2.d) v2: (two y-axes) plot and save figure of LWP from Radar-89GHz LWP and MWR IWV
fig, ax = pyLARDA.Transformations.plot_timeseries(Radar_LWP_PuntaRetr, rg_converter=True,
                                                  title=True, zlim=[-3, 3])
ax.set_ylim([0, 2200])  # limit for left y-axis
ax.set_ylabel('89 GHz Ch. LWP [g m^-2]', color='royalblue', fontweight='bold', fontsize=15)
ax_right = ax.twinx()  # IWV
ax_right.set_ylabel('MWR IWV [kg m^-2]', color='black', fontweight='bold', fontsize=15)
ax_right.set_ylim([0, iwv_max])  # replace with IWV later
ax_right.plot(dt_MWR, MWR_IWV['var'], color='black', linewidth=2.0)
ax_right.scatter(dt_disdro, disdro_RainFlag, marker='*', color='red')
# ax_right.set_xlabel([])
ax_right.tick_params(axis='both', which='both', right=True, top=True)
ax_right.tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
ax_right.tick_params(axis='both', which='minor', width=2, length=3)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0]))
ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=[12]))
ax.set_title(f'Punta Arenas {dt_begin:%Y%m%d} - {dt_end:%Y%m%d}', fontweight='bold', fontsize=15)
fig.savefig(name + 'LWP_IWV.png', dpi=250)

# 3.a) plot and save figure of disdrometer rainrate
fig, ax = pyLARDA.Transformations.plot_timeseries(disdro_RR, rg_converter=True,
                                                  title=True, zlim=[-3, 3])
ax.set_ylim([0, 10])
fig.savefig(name + system_disdro + 'rainrate.png', dpi=250)

# 3.b) plot and save figure of LIMRAD94 weather station rainrate
fig, ax = pyLARDA.Transformations.plot_timeseries(Radar_RR, rg_converter=True,
                                                  title=True, zlim=[-3, 3])
ax.set_ylim([0, 10])
fig.savefig(name + system_Limrad94 + 'rainrate.png', dpi=250)

# 3.c) plot and save figure of disdrometer rainrate AND LIMRAD94 weather station rainrate
fig, ax = pyLARDA.Transformations.plot_timeseries(disdro_RR, rg_converter=True,
                                                  title=True, zlim=[-3, 3])
ax.plot(dt_Limrad94, Radar_RR['var'], color='black', linewidth=1.0)
ax.legend(['disdrometer', 'LIMRAD94 weather station'])
ax.set_ylim([0, 10])
ax.set_xlim([dt_begin, dt_end])
fig.savefig(name + 'rainrate_both.png', dpi=250)

# 4. LIMRAD94 weather station surface T and RH
fig, ax = pyLARDA.Transformations.plot_timeseries(Radar_SurfT, rg_converter=True,
                                                  title=True, zlim=[-3, 3])
ax.set_ylim([0, T_max])  # limit for left y-axis
ax.set_ylabel('Surface T [°C]', color='royalblue', fontweight='bold', fontsize=15)
ax_right = ax.twinx()  # RH
ax_right.set_ylabel('Surface Rel.Hum. [%]', color='black', fontweight='bold', fontsize=15)
ax_right.set_ylim([0, RH_max])
ax_right.plot(dt_Limrad94, Radar_SurfRH['var'], color='black', linewidth=2.0)
ax_right.scatter(dt_disdro, disdro_RainFlag, marker='*', color='red')
# ax_right.set_xlabel([])
ax_right.tick_params(axis='both', which='both', right=True, top=True)
ax_right.tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
ax_right.tick_params(axis='both', which='minor', width=2, length=3)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0]))
ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=[12]))
ax.set_title(f'Punta Arenas {dt_begin:%Y%m%d} - {dt_end:%Y%m%d}', fontweight='bold', fontsize=15)
fig.savefig(name + 'surf_T_RH.png', dpi=250)

# example for subplots and two y-axes
# fig, ax = plt.subplots(2, 1, figsize=[12, 9]) # --> 2 rows, 1 col, upper plot [0], lower plot[1]

'''bsc['var_lims'] = [1e-7, 1e-3]
            depol['var_lims'] = [0, 0.3]

            ln1 = ax[0].plot(bsc['rg'], bsc_cpy[iT, :], color='royalblue', label=r'$\beta_{1064}$')
            ax[0].set_xlabel(f'range [{bsc["rg_unit"]}]', fontsize=font_size, fontweight=font_weight)
            ax[0].set_ylabel(f'att. bsc. [{bsc["var_unit"]}]', color='royalblue', fontsize=font_size, fontweight=font_weight)
            ax[0].set_yscale('log')
            ax[0].set_ylim(bsc['var_lims'])
            ax[0].set_xlim(plot_range)

            ax0 = ax[0].twinx()
            ax0.set_ylabel('depol', color='grey', fontsize=font_size, fontweight=font_weight)
            ln3 = ax0.plot(depol['rg'], depol_corrected, 'red', label=r'$\delta_{532}$ corrected')
            ln2 = ax0.plot(depol['rg'], depol_cpy[iT, :], 'grey', label=r'$\delta_{532}$')
            ax0.set_ylim([0, 0.5])'''
