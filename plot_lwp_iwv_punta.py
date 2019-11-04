#!/usr/bin/python3

import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import netCDF4 as nc

# path to local larda source code - no data needed locally
sys.path.append("C:\\Users\\Johannes\\Studium\\Hiwi_Kalesse\\larda3_v2\\larda_local\\larda")
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

# load data from server
begin_dt = datetime.datetime(2018, 12, 2, 20, 50, 30)
end_dt = datetime.datetime(2018, 12, 7, 23, 59, 59)

hatpro_lwp_debilt = larda.read("HATPRO", "LWP", [begin_dt, end_dt])
hatpro_iwv_debilt = larda.read("HATPRO", "IWV", [begin_dt, end_dt])
limrad94_lwp = larda.read("LIMRAD94", "LWP", [begin_dt, end_dt])
limrad94_surf_rel_hum = larda.read("LIMRAD94", "SurfRelHum", [begin_dt, end_dt])
parsivel_rainrate = larda.read("PARSIVEL", "rainrate", [begin_dt, end_dt])

# load local data
path = "./HATPRO-Punta_Arenas/"
hatpro_lwp_pa_ds = nc.Dataset(path+"MWR_PRO_PA_retrieval/ioppta_lac_mwr00_l2_clwvi_v01_20181202205030-20181207000000.nc")
hatpro_lwp_rpg_ds = nc.Dataset(path+"RPG-PA_retrieval/181202-181207.LWP.NC")
hatpro_iwv_pa_ds = nc.Dataset(path+"MWR_PRO_PA_retrieval/ioppta_lac_mwr00_l2_prw_v01_20181202205030-20181207000000.nc")
hatpro_iwv_rpg_ds = nc.Dataset(path+"RPG-PA_retrieval/181202-181207.IWV.NC")
# extract variables from nc files
# compared both time axis, rpg file hsa one more datapoint which is removed in the variable selection
nc_time = nc.num2date(hatpro_lwp_pa_ds.variables["time"][:], units="seconds since 1970-01-01 00:00:00 UTC",
                      calendar="standard")
parsivel_time = nc.num2date(parsivel_rainrate["ts"][:], units="seconds since 1970-01-01 00:00:00 UTC")
hatpro_lwp_pa = hatpro_lwp_pa_ds.variables["clwvi"][:] * 1e3  # convert to g / m^2
hatpro_lwp_rpg = hatpro_lwp_rpg_ds.variables["LWP_data"][:-1]  # g / m^2
hatpro_iwv_pa = hatpro_iwv_pa_ds.variables["prw"][:]  # kg / m^2
hatpro_iwv_rpg = hatpro_iwv_rpg_ds.variables["IWV_data"][:-1]  # kg / m^2
hatpro_rainflag_rpg = hatpro_lwp_rpg_ds.variables["rain_flag"][:-1]  # 0=no rain, 1=raining
hatpro_rainflag_rpg = hatpro_rainflag_rpg == 1

# plot time series
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
fig, ax = pyLARDA.Transformations.plot_timeseries(hatpro_lwp_debilt)
fig.savefig("test.png")
plt.close()
# plot liquid water path with rain flags
fig, ax = plt.subplots(figsize=(16, 9), dpi=600)
ax.plot_date(nc_time[:-1], hatpro_lwp_debilt["var"]*1e3, 'r-', label="DeBilt")
ax.plot_date(nc_time, hatpro_lwp_pa, 'b--', label="PuntaArenas")
ax.plot_date(nc_time, hatpro_lwp_rpg, 'g:', label="RPG_NeuralNetwork")
ax.set_ylim(top=max(hatpro_lwp_rpg[~hatpro_rainflag_rpg]))
ax.set_xlabel("Date")
ax.set_ylabel(r"Liquid Water Path [$g \, m^{-2}$]")
ax.set_title("HATPRO Liquid Water Path from different retrievals\n Dacapo Peso Punta Arenas")
ax.fill_between(x=nc_time, y1=max(hatpro_lwp_rpg), y2=0, where=hatpro_rainflag_rpg, alpha=0.7, color='grey',
                label='RPG rainflag')
ax.legend(title="Retrieval", loc='upper center', ncol=1)
fig.savefig("MWR_lwp_punta.png")
plt.close()

print("end of plotting")
