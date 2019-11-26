#!/usr/bin/python3

import os
import sys
import functions_jr as jr
import datetime
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import netCDF4 as nc

# path to local larda source code - no data needed locally
sys.path.append("C:\\Users\\Johannes\\Studium\\Hiwi_Kalesse\\larda")
print(os.getcwd())
print(sys.executable)

import pyLARDA
import pyLARDA.helpers as h
# import pyLARDA.Transformations as pLTransf

# # optionally setup logging
# import logging
# logger = logging.getLogger("ipy")
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())

larda = pyLARDA.LARDA('remote', uri='http://larda.tropos.de/larda3')
larda.connect('lacros_dacapo', build_lists=True)

########################################################################################################################
# Loading Data
########################################################################################################################
# load data from server
begin_dt = datetime.datetime(2018, 12, 2, 20, 50, 30)
end_dt = datetime.datetime(2018, 12, 7, 23, 59, 59)

# hatpro_lwp_pa = larda.read("HATPRO", "LWP", [begin_dt, end_dt])  # kg / m^2
hatpro_flag_pa = larda.read("HATPRO", "flag", [begin_dt, end_dt])  # flags
limrad94_lwp = larda.read("LIMRAD94", "LWP", [begin_dt, end_dt])  # g / m^2
limrad94_surf_rel_hum = larda.read("LIMRAD94", "SurfRelHum", [begin_dt, end_dt])
parsivel_rainrate = larda.read("PARSIVEL", "rainrate", [begin_dt, end_dt])  # mm / h

# load local data
path = "./HATPRO-Punta_Arenas/"
hatpro_lwp_debilt_ds = nc.Dataset(path+"MWR_PRO_DeBilt_retrieval/ioppta_lac_mwr00_l2_clwvi_v00_20181202205030-20181207235959.nc")
hatpro_iwv_debilt_ds = nc.Dataset(path+"MWR_PRO_DeBilt_retrieval/ioppta_lac_mwr00_l2_prw_v00_20181202205030-20181207235959.nc")
hatpro_lwp_pa_ds = nc.Dataset(path+"MWR_PRO_PA_retrieval/ioppta_lac_mwr00_l2_clwvi_v01_20181202205030-20181207000000.nc")
hatpro_lwp_rpg_ds = nc.Dataset(path+"RPG-PA_retrieval/181202-181207.LWP.NC")
hatpro_iwv_pa_ds = nc.Dataset(path+"MWR_PRO_PA_retrieval/ioppta_lac_mwr00_l2_prw_v01_20181202205030-20181207000000.nc")
hatpro_iwv_rpg_ds = nc.Dataset(path+"RPG-PA_retrieval/181202-181207.IWV.NC")

# extract variables from nc files
# compared both time axis, rpg file has one more datapoint which is removed in the variable selection
nc_time = nc.num2date(hatpro_lwp_pa_ds.variables["time"][:], units="seconds since 1970-01-01 00:00:00 UTC",
                      calendar="standard")
# get parsivel and limrad time because of different sampling intervals
parsivel_time = nc.num2date(parsivel_rainrate["ts"][:], units="seconds since 1970-01-01 00:00:00 UTC")
limrad94_time = nc.num2date(limrad94_lwp["ts"][:], units="seconds since 1970-01-01 00:00:00 UTC")
hatpro_lwp_pa = hatpro_lwp_pa_ds.variables["clwvi"][:] * 1e3  # convert to g / m^2
hatpro_lwp_rpg = hatpro_lwp_rpg_ds.variables["LWP_data"][:-1]  # g / m^2
hatpro_lwp_debilt = hatpro_lwp_debilt_ds.variables["clwvi"][:] * 1e3  # convert to g / m^2
hatpro_iwv_pa = hatpro_iwv_pa_ds.variables["prw"][:]  # kg / m^2
hatpro_iwv_rpg = hatpro_iwv_rpg_ds.variables["IWV_data"][:-1]  # kg / m^2
hatpro_iwv_debilt = hatpro_iwv_debilt_ds.variables["prw"][:]  # kg / m^2
# rainflags
hatpro_rainflag_rpg = hatpro_lwp_rpg_ds.variables["rain_flag"][:-1] == 1  # 0=no rain, 1=raining
hatpro_rainflag_pa = hatpro_lwp_pa_ds.variables["flag"][:] == 8  # 8=rain
parsivel_rainflag = parsivel_rainrate["var"][:] > 0
# quality flags
# check for flags except for rain flags, False=some error, True=all good
hatpro_flags_pa = np.isin(hatpro_lwp_pa_ds.variables["flag"][:], [0, 8])
hatpro_flags_all_pa = (hatpro_lwp_pa_ds.variables["flag"][:] == 0).mask

####################################################################################
# plotting section
####################################################################################

# plot time series
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
# fig, ax = pyLARDA.Transformations.plot_timeseries(hatpro_lwp_debilt)
# fig.savefig("test.png")
# plt.close()

# plot liquid water path with rain flags
fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
ax.plot_date(nc_time, hatpro_lwp_debilt, 'r-', label="DeBilt-Radiosonde")
ax.plot_date(nc_time, hatpro_lwp_pa, 'b--', label="Punta Arenas-ERA5")
ax.plot_date(nc_time, hatpro_lwp_rpg, 'g:', label="RPG Punta Arenas Neural Network")
ax.plot_date(limrad94_time, limrad94_lwp["var"], 'k-.', label="LIMRAD94-89GHz")
ax.set_ylim(top=max(hatpro_lwp_rpg[~hatpro_rainflag_rpg]))
ax.set_xlabel("Date")
ax.set_ylabel(r"Liquid Water Path [$g \, m^{-2}$]")
ax.set_title("Liquid Water Path from different HATPRO retrievals and LIMRAD94\n Dacapo Peso - Punta Arenas")
ax.fill_between(x=nc_time, y1=max(hatpro_lwp_rpg), y2=0, where=hatpro_rainflag_rpg, alpha=0.7, color='deepskyblue',
                label='RPG rainflag')
ax.fill_between(x=parsivel_time, y1=max(hatpro_lwp_rpg), y2=0, where=parsivel_rainflag, alpha=0.7, color='orangered',
                label='PARSIVEL rainflag')
ax.fill_between(x=nc_time, y1=max(hatpro_lwp_rpg), y2=0, where=~hatpro_flags_pa, alpha=0.7, color='limegreen',
                label='Other Flag')
# TODO: Add weather station from LIMRAD lv1 files
ax.legend(title="Retrieval", loc='upper center', ncol=1)
fig.savefig("MWR_lwp_punta.png")
plt.close()

# zoom in
fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
ax.plot_date(nc_time, hatpro_lwp_debilt, 'r-', label="DeBilt-Radiosonde")
ax.plot_date(nc_time, hatpro_lwp_pa, 'b--', label="Punta Arenas-ERA5")
ax.plot_date(nc_time, hatpro_lwp_rpg, 'g:', label="RPG Punta Arenas Neural Network")
ax.plot_date(limrad94_time, limrad94_lwp["var"], 'k-.', label="LIMRAD94-89GHz")
ax.set_ylim(top=1500)
ax.set_xlim(left=datetime.datetime(2018, 12, 6), right=datetime.datetime(2018, 12, 7))
ax.set_xlabel("Datetime [$UTC$]")
ax.set_ylabel(r"Liquid Water Path [$g \, m^{-2}$]")
ax.set_title("Liquid Water Path from different HATPRO retrievals and LIMRAD94\n Dacapo Peso - Punta Arenas")
ax.fill_between(x=nc_time, y1=max(hatpro_lwp_rpg), y2=0, where=hatpro_rainflag_rpg, alpha=0.7, color='deepskyblue',
                label='RPG rainflag')
ax.fill_between(x=parsivel_time, y1=max(hatpro_lwp_rpg), y2=0, where=parsivel_rainflag, alpha=0.7, color='orangered',
                label='PARSIVEL rainflag')
ax.fill_between(x=nc_time, y1=max(hatpro_lwp_rpg), y2=0, where=~hatpro_flags_pa, alpha=0.7, color='limegreen',
                label='Other Flag')
# TODO: Add weather station from LIMRAD lv1 files
ax.legend(title="Retrieval", loc='upper right', ncol=1)
fig.savefig("MWR_lwp_punta_zoom.png")
plt.close()

# zoom in 2
fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
ax.plot_date(nc_time, hatpro_lwp_debilt, 'r-', label="DeBilt-Radiosonde")
ax.plot_date(nc_time, hatpro_lwp_pa, 'b--', label="Punta Arenas-ERA5")
ax.plot_date(nc_time, hatpro_lwp_rpg, 'g:', label="RPG Punta Arenas Neural Network")
ax.plot_date(limrad94_time, limrad94_lwp["var"], 'k-.', label="LIMRAD94-89GHz")
ax.set_ylim(top=500)
ax.set_xlim(left=datetime.datetime(2018, 12, 6, 12), right=datetime.datetime(2018, 12, 7))
ax.set_xlabel("Datetime [$UTC$]")
ax.set_ylabel(r"Liquid Water Path [$g \, m^{-2}$]")
ax.set_title("Liquid Water Path from different HATPRO retrievals and LIMRAD94\n Dacapo Peso - Punta Arenas")
ax.fill_between(x=nc_time, y1=max(hatpro_lwp_rpg), y2=0, where=hatpro_rainflag_rpg, alpha=0.7, color='deepskyblue',
                label='RPG rainflag')
ax.fill_between(x=parsivel_time, y1=max(hatpro_lwp_rpg), y2=0, where=parsivel_rainflag, alpha=0.7, color='orangered',
                label='PARSIVEL rainflag')
ax.fill_between(x=nc_time, y1=max(hatpro_lwp_rpg), y2=0, where=~hatpro_flags_pa, alpha=0.7, color='limegreen',
                label='Other Flag')
# TODO: Add weather station from LIMRAD lv1 files
ax.legend(title="Retrieval", loc='upper right', ncol=1)
fig.savefig("MWR_lwp_punta_zoom2.png")
plt.close()

# plot time series of iwv with rainflags
fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
ax.plot_date(nc_time, hatpro_iwv_debilt, 'r-', label="DeBilt-Radiosonde")
ax.plot_date(nc_time, hatpro_iwv_pa, 'b--', label="Punta Arenas-ERA5")
ax.plot_date(nc_time, hatpro_iwv_rpg, 'g:', label="RPG Punta Arenas Neural Network")
# ax.plot_date(limrad94_time, limrad94_lwp["var"], 'k-.', label="LIMRAD94")
ax.set_ylim(top=max(hatpro_iwv_rpg[~hatpro_rainflag_rpg]))
ax.set_xlabel("Date")
ax.set_ylabel(r"Integrated Water Vapor [$kg \, m^{-2}$]")
ax.set_title("Integrated Water Vapor from different HATPRO retrievals\n Dacapo Peso - Punta Arenas")
ax.fill_between(x=nc_time, y1=max(hatpro_iwv_rpg), y2=0, where=hatpro_rainflag_rpg, alpha=0.7, color='deepskyblue',
                label='RPG rainflag')
ax.fill_between(x=parsivel_time, y1=max(hatpro_iwv_rpg), y2=0, where=parsivel_rainflag, alpha=0.7, color='orangered',
                label='PARSIVEL rainflag')
ax.fill_between(x=nc_time, y1=max(hatpro_iwv_rpg), y2=0, where=~hatpro_flags_pa, alpha=0.7, color='limegreen',
                label='Other Flag')
# TODO: Add weather station from LIMRAD lv1 files
ax.legend(title="Retrieval", loc='upper center', ncol=1)
fig.savefig("MWR_iwv_punta.png")
plt.close()

# # zoom in
# fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
# ax.plot_date(nc_time[:-1], hatpro_iwv_debilt["var"]*1e3, 'r-', label="DeBilt")
# ax.plot_date(nc_time, hatpro_iwv_pa, 'b--', label="PuntaArenas")
# ax.plot_date(nc_time, hatpro_iwv_rpg, 'g:', label="RPG Punta Arenas Neural Network")
# # ax.plot_date(limrad94_time, limrad94_lwp["var"], 'k-.', label="LIMRAD94")
# ax.set_ylim(top=1500)
# ax.set_xlim(left=datetime.datetime(2018, 12, 6), right=datetime.datetime(2018, 12, 7))
# ax.set_xlabel("Datetime [$UTC$]")
# ax.set_ylabel(r"Integrated Water Vapor [$kg/m^{-2}$]")
# ax.set_title("HATPRO Integrated Water Vapor from different retrievals\n Dacapo Peso - Punta Arenas")
# ax.fill_between(x=nc_time, y1=max(hatpro_lwp_rpg), y2=0, where=hatpro_rainflag_rpg, alpha=0.7, color='deepskyblue',
#                 label='RPG rainflag')
# ax.fill_between(x=parsivel_time, y1=max(hatpro_lwp_rpg), y2=0, where=parsivel_rainflag, alpha=0.7, color='orangered',
#                 label='PARSIVEL rainflag')
# # TODO: Add weather station from LIMRAD lv1 files
# ax.legend(title="Retrieval", loc='upper right', ncol=1)
# fig.savefig("MWR_iwv_punta_zoom.png")
# plt.close()

# scatterplots with regression line non rainy pixels
# DeBilt vs Punta
# regression line
outfiles = ["MWR_lwp_comp_debilt-punta.png", "MWR_lwp_comp_rpg-punta.png", "MWR_iwv_comp_debilt-punta.png",
            "MWR_iwv_comp_rpg-punta.png"]
x_ls = [hatpro_lwp_debilt[~hatpro_rainflag_rpg], hatpro_lwp_rpg[~hatpro_rainflag_rpg],
        hatpro_iwv_debilt[~hatpro_rainflag_rpg], hatpro_iwv_rpg[~hatpro_rainflag_rpg]]
y_ls = [hatpro_lwp_pa[~hatpro_rainflag_rpg], hatpro_lwp_pa[~hatpro_rainflag_rpg],
        hatpro_iwv_pa[~hatpro_rainflag_rpg], hatpro_iwv_pa[~hatpro_rainflag_rpg]]
y_labels = [r"LWP Punta Arenas-ERA5 Retrieval [$g \, m^{-2}$]", r"LWP Punta Arenas-ERA5 Retrieval [$g \, m^{-2}$]",
            r"IWV Punta Arenas-ERA5 Retrieval [$kg \, m^{-2}$]", r"IWV Punta Arenas-ERA5 Retrieval [$kg \, m^{-2}$]"]
x_labels = [r"LWP DeBilt-Radiosonde Retrieval [$g \, m^{-2}$]", r"LWP RPG Retrieval [$g \, m^{-2}$]",
            r"IWV DeBilt-Radiosonde Retrieval [$kg \, m^{-2}$]", r"IWV RPG Retrieval [$kg \, m^{-2}$]"]
titles = ["Comparison of Liquid Water Path from different HATPRO retrievals-non rainy pixels (HATPRO)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Liquid Water Path from different HATPRO retrievals-non rainy pixels (HATPRO)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Integrated Water Vapor from different HATPRO retrievals-non rainy pixels (HATPRO)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Integrated Water Vapor from different HATPRO retrievals-non rainy pixels (HATPRO)"
          "\n Dacapo Peso - Punta Arenas"]
i = 0
for y, y_label, x, x_label, title in zip(y_ls, y_labels, x_ls, x_labels, titles):
    b, m = np.polynomial.polynomial.polyfit(x, y, 1)
    r_squared = jr.polyfit(x, y, 1)['determination']
    fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
    ax.plot(x, y, 'k.')
    ax.plot(x, b + m * x, 'g-', label=f"1st degree polynomial fit\n$y={b:.2f}+{m:.2f}*x$\n$R^2={r_squared:.4f}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.savefig(outfiles[i])
    plt.close()
    i += 1

# plot only non flagged pixels
outfiles = ["MWR_lwp_comp_noflag_debilt-punta.png", "MWR_lwp_comp_noflag_rpg-punta.png",
            "MWR_iwv_comp_noflag_debilt-punta.png", "MWR_iwv_comp_noflag_rpg-punta.png"]
x_ls = [hatpro_lwp_debilt[hatpro_flags_all_pa], hatpro_lwp_rpg[hatpro_flags_all_pa],
        hatpro_iwv_debilt[hatpro_flags_all_pa], hatpro_iwv_rpg[hatpro_flags_all_pa]]
y_ls = [hatpro_lwp_pa[hatpro_flags_all_pa], hatpro_lwp_pa[hatpro_flags_all_pa],
        hatpro_iwv_pa[hatpro_flags_all_pa], hatpro_iwv_pa[hatpro_flags_all_pa]]
y_labels = [r"LWP Punta Arenas-ERA5 Retrieval [$g \, m^{-2}$]", r"LWP Punta Arenas-ERA5 Retrieval [$g \, m^{-2}$]",
            r"IWV Punta Arenas-ERA5 Retrieval [$kg \, m^{-2}$]", r"IWV Punta Arenas-ERA5 Retrieval [$kg \, m^{-2}$]"]
x_labels = [r"LWP DeBilt-Radiosonde Retrieval [$g \, m^{-2}$]", r"LWP RPG Retrieval [$g \, m^{-2}$]",
            r"IWV DeBilt-Radiosonde Retrieval [$kg \, m^{-2}$]", r"IWV RPG Retrieval [$kg \, m^{-2}$]"]
titles = ["Comparison of Liquid Water Path from different HATPRO retrievals-non flagged pixels (HATPRO)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Liquid Water Path from different HATPRO retrievals-non flagged pixels (HATPRO)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Integrated Water Vapor from different HATPRO retrievals-non flagged pixels (HATPRO)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Integrated Water Vapor from different HATPRO retrievals-non flagged pixels (HATPRO)"
          "\n Dacapo Peso - Punta Arenas"]
i = 0
for y, y_label, x, x_label, title in zip(y_ls, y_labels, x_ls, x_labels, titles):
    b, m = np.polynomial.polynomial.polyfit(x, y, 1)
    r_squared = jr.polyfit(x, y, 1)['determination']
    fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
    ax.plot(x, y, 'k.')
    ax.plot(x, b + m * x, 'g-', label=f"1st degree polynomial fit\n$y={b:.2f}+{m:.2f}*x$\n$R^2={r_squared:.4f}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.savefig(outfiles[i])
    plt.close()
    i += 1

# set IWV and LWP values for disdrometer rainy pxl to NaN
# a) interpolate disdro time resolution to MWR time resolution + set MWR-IWV and MWR-LWP at disdro rainy profiles to NaN
f = interpolate.interp1d(parsivel_rainrate['ts'], parsivel_rainrate['var'], kind='nearest')
disdro_RR_ip2mwr = f(hatpro_flag_pa['ts'])  # interpolate Disdro RR to MWR time values

disdro_RainFlag_ip2mwr = disdro_RR_ip2mwr > 0  # Rain Flag at MWR time steps

# b) interpolate disdro time resolution to LIMRAD94 time resolution and set 89GHz LWP at rainy profiles to NaN
f = interpolate.interp1d(parsivel_rainrate['ts'], parsivel_rainrate['var'], kind='nearest', bounds_error=False,
                         fill_value=-1)
disdro_RR_ip2Limrad94 = f(limrad94_lwp['ts'])  # interpolate Disdro RR to Limrad94 time steps
disdro_RainFlag_ip2Limrad94 = disdro_RR_ip2Limrad94 > 0  # Rain Flag at Limrad94 time steps


# plot only non rainy pixels
outfiles = ["MWR_lwp_comp_norain_disdro_debilt-punta.png", "MWR_lwp_comp_norain_disdro_rpg-punta.png",
            "MWR_iwv_comp_norain_disdro_debilt-punta.png", "MWR_iwv_comp_norain_disdro_rpg-punta.png"]
x_ls = [hatpro_lwp_debilt[:-1][~disdro_RainFlag_ip2mwr], hatpro_lwp_rpg[:-1][~disdro_RainFlag_ip2mwr],
        hatpro_iwv_debilt[:-1][~disdro_RainFlag_ip2mwr], hatpro_iwv_rpg[:-1][~disdro_RainFlag_ip2mwr]]
y_ls = [hatpro_lwp_pa[:-1][~disdro_RainFlag_ip2mwr], hatpro_lwp_pa[:-1][~disdro_RainFlag_ip2mwr],
        hatpro_iwv_pa[:-1][~disdro_RainFlag_ip2mwr], hatpro_iwv_pa[:-1][~disdro_RainFlag_ip2mwr]]
y_labels = [r"LWP Punta Arenas-ERA5 Retrieval [$g \, m^{-2}$]", r"LWP Punta Arenas-ERA5 Retrieval [$g \, m^{-2}$]",
            r"IWV Punta Arenas-ERA5 Retrieval [$kg \, m^{-2}$]", r"IWV Punta Arenas-ERA5 Retrieval [$kg \, m^{-2}$]"]
x_labels = [r"LWP DeBilt-Radiosonde Retrieval [$g \, m^{-2}$]", r"LWP RPG Retrieval [$g \, m^{-2}$]",
            r"IWV DeBilt-Radiosonde Retrieval [$kg \, m^{-2}$]", r"IWV RPG Retrieval [$kg \, m^{-2}$]"]
titles = ["Comparison of Liquid Water Path from different HATPRO retrievals-non rainy pixels (Parsivel)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Liquid Water Path from different HATPRO retrievals-non rainy pixels (Parsivel)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Integrated Water Vapor from different HATPRO retrievals-non rainy pixels (Parsivel)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Integrated Water Vapor from different HATPRO retrievals-non rainy pixels (Parsivel)"
          "\n Dacapo Peso - Punta Arenas"]
i = 0
for y, y_label, x, x_label, title in zip(y_ls, y_labels, x_ls, x_labels, titles):
    b, m = np.polynomial.polynomial.polyfit(x, y, 1)
    r_squared = jr.polyfit(x, y, 1)['determination']
    fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
    ax.plot(x, y, 'k.')
    ax.plot(x, b + m * x, 'g-', label=f"1st degree polynomial fit\n$y={b:.2f}+{m:.2f}*x$\n$R^2={r_squared:.4f}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.savefig(outfiles[i])
    plt.close()
    i += 1

# zoom in to 500
# plot only non flagged pixels
outfiles = ["MWR_lwp_comp_noflag_debilt-punta_zoom.png", "MWR_lwp_comp_noflag_rpg-punta_zoom.png",
            "MWR_iwv_comp_noflag_debilt-punta_zoom.png", "MWR_iwv_comp_noflag_rpg-punta_zoom.png"]
x_ls = [hatpro_lwp_debilt[:-1][~disdro_RainFlag_ip2mwr], hatpro_lwp_rpg[:-1][~disdro_RainFlag_ip2mwr],
        hatpro_iwv_debilt[:-1][~disdro_RainFlag_ip2mwr], hatpro_iwv_rpg[:-1][~disdro_RainFlag_ip2mwr]]
y_ls = [hatpro_lwp_pa[:-1][~disdro_RainFlag_ip2mwr], hatpro_lwp_pa[:-1][~disdro_RainFlag_ip2mwr],
        hatpro_iwv_pa[:-1][~disdro_RainFlag_ip2mwr], hatpro_iwv_pa[:-1][~disdro_RainFlag_ip2mwr]]
y_labels = [r"LWP Punta Arenas-ERA5 Retrieval [$g \, m^{-2}$]", r"LWP Punta Arenas-ERA5 Retrieval [$g \, m^{-2}$]",
            r"IWV Punta Arenas-ERA5 Retrieval [$kg \, m^{-2}$]", r"IWV Punta Arenas-ERA5 Retrieval [$kg \, m^{-2}$]"]
x_labels = [r"LWP DeBilt-Radiosonde Retrieval [$g \, m^{-2}$]", r"LWP RPG Retrieval [$g \, m^{-2}$]",
            r"IWV DeBilt-Radiosonde Retrieval [$kg \, m^{-2}$]", r"IWV RPG Retrieval [$kg \, m^{-2}$]"]
titles = ["Comparison of Liquid Water Path from different HATPRO retrievals-non rainy pixels (Parsivel)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Liquid Water Path from different HATPRO retrievals-non rainy pixels (Parsivel)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Integrated Water Vapor from different HATPRO retrievals-non rainy pixels (Parsivel)"
          "\n Dacapo Peso - Punta Arenas",
          "Comparison of Integrated Water Vapor from different HATPRO retrievals-non rainy pixels (Parsivel)"
          "\n Dacapo Peso - Punta Arenas"]
i = 0
for y, y_label, x, x_label, title in zip(y_ls, y_labels, x_ls, x_labels, titles):
    b, m = np.polynomial.polynomial.polyfit(x, y, 1)
    r_squared = jr.polyfit(x, y, 1)['determination']
    fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
    ax.plot(x, y, 'k.')
    ax.plot(x, b + m * x, 'g-', label=f"1st degree polynomial fit\n$y={b:.2f}+{m:.2f}*x$\n$R^2={r_squared:.4f}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(top=500)
    ax.set_xlim(right=500)
    ax.set_title(title)
    ax.legend()
    fig.savefig(outfiles[i])
    plt.close()
    i += 1

# plot histogram liquid water path
fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
ax.hist([hatpro_lwp_debilt[~hatpro_rainflag_rpg], hatpro_lwp_pa[~hatpro_rainflag_rpg],
         hatpro_lwp_rpg[~hatpro_rainflag_rpg]],
        bins=30,
        density=True,
        label=["DeBilt-Radiosonde", "Punta Arenas-ERA5", "RPG Punta Arenas Neural Network"])
# ax.set_xlim(right=1500)
# ax.set_ylim(top=4e-3)
ax.set_yscale('log')
ax.set_ylabel("Probability")
ax.set_xlabel(r"Liquid Water Path [$g \, m^{-2}$]")
# ax.yaxis.set_major_formatter(FormatStrFormatter('%2.1e'))
ax.set_title("HATPRO Probability Density Function of Liquid Water Path\n for Different Retrievals ")
ax.legend(title="Retrieval")
fig.tight_layout()
fig.savefig("MWR_lwp_hist.png")
plt.close()

# plot histogram integrated water vapor
fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
ax.hist([hatpro_iwv_debilt[~hatpro_rainflag_rpg], hatpro_iwv_pa[~hatpro_rainflag_rpg],
         hatpro_iwv_rpg[~hatpro_rainflag_rpg]],
        bins=30,
        density=True,
        label=["DeBilt-Radiosonde", "Punta Arenas-ERA5", "RPG Punta Arenas Neural Network"])
# ax.set_xlim(right=1500)
# ax.set_ylim(top=4e-3)
ax.set_yscale('log')
ax.set_ylabel("Probability")
ax.set_xlabel(r"Integrated Water Vapor [$kg \, m^{-2}$]")
# ax.yaxis.set_major_formatter(FormatStrFormatter('%2.1e'))
ax.set_title("HATPRO Probability Density Function of Integrated Water Vapor\n for Different Retrievals ")
ax.legend(title="Retrieval")
fig.tight_layout()
fig.savefig("MWR_iwv_hist.png")
plt.close()
print("end of plotting")

