#!/usr/bin/python3

import sys

# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pyLARDA
import pyLARDA.helpers as h
import datetime
import numpy as np

import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

# define plot path
plot_path = "/projekt1/remsens/work/jroettenbacher/plots"
# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

# gather command line arguments
method_name, args, kwargs = h._method_info_from_argv(sys.argv)
# gather argument date
if 'date_begin' in kwargs:
    date_begin = str(kwargs['date_begin'])
    begin_dt = datetime.datetime.strptime(date_begin + ' 00:00:05', '%Y%m%d %H:%M:%S')
else:
    begin_dt = datetime.datetime(2020, 2, 5, 0, 0, 5)

if 'date_end' in kwargs:
    date_end = str(kwargs['date_end'])
    end_dt = datetime.datetime.strptime(date_end + ' 23:59:55', '%Y%m%d %H:%M:%S')
else:
    end_dt = datetime.datetime(2020, 2, 5, 23, 59, 55)

if 'plot_range' in kwargs:
    plot_range = [0, int(kwargs['plot_range'])]
else:
    plot_range = [0, 3000]

#  read in moments
system = "LIMRAD94_cn_input"
radar_Z = larda.read(system, "Ze", [begin_dt, end_dt], plot_range)
# mask values = -999
radar_Z["var"] = np.ma.masked_where(radar_Z["var"] == -999, radar_Z["var"])
# overwrite mask in larda container -> does not change plot output
radar_Z["mask"] = radar_Z["var"].mask
radar_MDV = larda.read(system, "Vel", [begin_dt, end_dt], plot_range)
# overwrite mask in larda container
radar_MDV["mask"] = radar_Z["var"].mask
radar_sw = larda.read(system, "sw", [begin_dt, end_dt], plot_range)
# overwrite mask in larda container
radar_sw["mask"] = radar_Z["var"].mask
# radar_vel = larda.read(system, "vm", [begin_dt, end_dt], plot_range)
# radar_Inc_El = larda.read(system, "Inc_El", [begin_dt, end_dt])
# radar_Inc_ElA = larda.read(system, "Inc_ElA", [begin_dt, end_dt])
# radar_LDR = larda.read(system, "ldr", [begin_dt, end_dt], plot_range)
cloud_bases_tops = larda.read(system, "cloud_bases_tops", [begin_dt, end_dt], plot_range)
dt_list = np.asarray([datetime.datetime.utcfromtimestamp(time) for time in cloud_bases_tops['ts']])
range_list = cloud_bases_tops['rg']/1000.0
var = cloud_bases_tops['var']
# mask -999 = fill value
var = np.ma.masked_where(var == -999, var)

name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_preliminary_{plot_range[1] / 1000:.0f}km_cloudnet_input'

# plot Roll and pitch
# fig, ax = pyLARDA.Transformations.plot_timeseries(radar_Inc_El)
# formatted_datetime = (h.ts_to_dt(radar_Inc_El['ts'][0])).strftime("%Y-%m-%d")
# if not (h.ts_to_dt(radar_Inc_El['ts'][0])).strftime("%d") == (h.ts_to_dt(radar_Inc_El['ts'][-1])).strftime("%d"):
#     formatted_datetime = formatted_datetime + '-' + (h.ts_to_dt(radar_Inc_El['ts'][-1])).strftime("%d")
# ax.set_title(radar_Inc_El['paraminfo']['location'] + ', ' + formatted_datetime, fontsize=20)
# fig.savefig(f"{name}_Inc_El.png", dpi=250)
radar_Z['var_unit'] = 'dBZ'
radar_Z['colormap'] = 'jet'
fig, _ = pyLARDA.Transformations.plot_timeheight(radar_Z, range_interval=plot_range, rg_converter=True, title=True,
                                                 z_converter='lin2z')
fig.savefig(name + '_Z.png', dpi=250)
print(f'figure saved :: {name}_Z.png')

fig, ax = pyLARDA.Transformations.plot_timeheight(radar_Z, range_interval=plot_range, rg_converter=True, title=True,
                                                  z_converter='lin2z')
# create a binary color map for bases and tops
cmap = ListedColormap(["darkgreen", "darkmagenta"])
ax.pcolormesh(matplotlib.dates.date2num(dt_list[:]), range_list[:], np.transpose(var[:, :]),
              label='cloud bases and tops', cmap=cmap)
legend_elements = [Patch(facecolor='darkgreen', label='Bases'),
                   Patch(facecolor='darkmagenta', label='Tops')]
ax.legend(handles=legend_elements, title="Hydrometeor")
fig.savefig(name + '_cbt_Z.png', dpi=250)
print(f'figure saved :: {name}_cbt_Z.png')
#
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_MDV, rg_converter=True, title=True)
# fig.savefig(name + '_MDV.png', dpi=250)
# print(f'figure saved :: {name}_MDV.png')
# #
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_sw, rg_converter=True, title=True)
# fig.savefig(name + '_width.png', dpi=250)
# print(f'figure saved :: {name}_width.png')
#
# radar_LDR['var_lims'] = [-30, 0]
# radar_LDR['colormap'] = 'jet'
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_LDR, rg_converter=True, title=True)
# fig.savefig(name+'_LDR.png', dpi=250)
# print(f'figure saved :: {name}_LDR.png')
#
# radar_ZDR['var_lims'] = [-0.2, 1]
# radar_ZDR['colormap'] = 'jet'
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_ZDR, rg_converter=True, title=True)
# fig.savefig(name+'_ZDR.png', dpi=250)
# print(f'figure saved :: {name}_ZDR.png')
#
# radar_RHV['var_lims'] = [0.85, 1]
# radar_RHV['colormap'] = 'jet'
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_RHV, rg_converter=True, title=True)
# fig.savefig(name+'_RHV.png', dpi=250)
# print(f'figure saved :: {name}_RHV.png')
#
# radar_PhiDP['var_lims'] = [0, 5]
# radar_PhiDP['colormap'] = 'jet'
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_PhiDP, rg_converter=True, title=True)
# fig.savefig(name+'_PhiDP.png', dpi=250)
# print(f'figure saved :: {name}_PhiDP.png')
