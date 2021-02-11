#!/usr/bin/python3

import sys
# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import pyLARDA
import pyLARDA.helpers as h
import datetime
import numpy as np

import logging


def toC(datalist):
    return datalist[0]['var'] - 273.15, datalist[0]['mask']


log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

# gather command line arguments
method_name, args, kwargs = h._method_info_from_argv(sys.argv)
# gather argument date
if 'date' in kwargs:
    date = str(kwargs['date'])
    begin_dt = datetime.datetime.strptime(date + ' 00:00:05', '%Y%m%d %H:%M:%S')
    end_dt = datetime.datetime.strptime(date + ' 23:59:55', '%Y%m%d %H:%M:%S')
else:
    begin_dt = datetime.datetime(2020, 2, 15, 0, 0, 5)
    end_dt = datetime.datetime(2020, 2, 15, 23, 59, 55)

if 'plot_range' in kwargs:
    plot_range = [0, int(kwargs['plot_range'])]
else:
    plot_range = [0, 12000]

#  read in moments
system = "LIMRAD94"
radar_Z   = larda.read(system, "Ze", [begin_dt, end_dt], plot_range)
radar_MDV = larda.read(system, "VEL", [begin_dt, end_dt], plot_range)
# radar_sw = larda.read(system, "sw", [begin_dt, end_dt], plot_range)
# radar_LDR = larda.read(system, "ldr", [begin_dt, end_dt], plot_range)
# radar_ZDR = larda.read(system, "ZDR", [begin_dt, end_dt], plot_range)
# radar_RHV = larda.read(system, "RHV", [begin_dt, end_dt], plot_range)
# radar_PhiDP = larda.read(system, "PhiDP", [begin_dt, end_dt], plot_range)
# radar_SurfWS = larda.read(system, "SurfWS", [begin_dt, end_dt])

# read in temperature and cloudnet classification
cloudnet = "CLOUDNET_LIMRAD"
T = larda.read(cloudnet, "T", [begin_dt, end_dt], plot_range)
classification = larda.read(cloudnet, "target_classification", [begin_dt, end_dt], plot_range)
T = pyLARDA.Transformations.combine(toC, [T], {'var_unit': "C"})
contour_T = {'data': T, 'levels': np.arange(-40, 16, 5)}

location = radar_MDV["paraminfo"]["location"]
name = f'plots/{location}_{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1]/1000:.0f}km'

# fig, _ = pyLARDA.Transformations.plot_timeseries(radar_SurfWS, title=True, z_converter='lin2z')
# fig.savefig(name+'_SurfWS.png', dpi=250)
# print(f'figure saved :: {name}_SurfWS.png')

fig, ax = pyLARDA.Transformations.plot_timeheight2(classification, contour=contour_T)
fig.savefig(name+'_class.png', dpi=250)
print(f'figure saved :: {name}_class.png')

radar_Z['var_unit'] = 'dBZ'
fig, _ = pyLARDA.Transformations.plot_timeheight2(radar_Z, rg_converter=True, z_converter='lin2z')
fig.savefig(name+'_Z.png', dpi=250)
print(f'figure saved :: {name}_Z.png')

fig, _ = pyLARDA.Transformations.plot_timeheight2(radar_MDV, range_interval=[6000, 9000])
fig.savefig(name+'_MDV.png', dpi=250)
print(f'figure saved :: {name}_MDV.png')
#
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_sw, rg_converter = True, title = True)
# fig.savefig(name + '_width.png', dpi = 250)
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