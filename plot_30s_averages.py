#!/usr/bin/python3

import sys

# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import matplotlib

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
    plot_range = [0, 12000]

#  read in moments
system = "LIMRAD94_30s"
radar_Z = larda.read(system, "Ze", [begin_dt, end_dt], plot_range)
# mask values = -999
radar_Z["var"] = np.ma.masked_where(radar_Z["var"] == -999, radar_Z["var"])
# overwrite mask in larda container -> does not change plot output
radar_Z["mask"] = radar_Z["var"].mask
name = f'plots/' \
       f'{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_preliminary_{plot_range[1] / 1000:.0f}km_30s'

radar_Z['var_unit'] = 'dBZ'
radar_Z['colormap'] = 'jet'
fig, _ = pyLARDA.Transformations.plot_timeheight(radar_Z, range_interval=plot_range, rg_converter=True, title=True,
                                                 z_converter='lin2z')
fig.savefig(name + '_Z.png', dpi=250)
print(f'figure saved :: {name}_Z.png')
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
