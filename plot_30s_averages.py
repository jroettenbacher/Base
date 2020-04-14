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
    begin_dt = datetime.datetime(2020, 2, 5, 12, 0, 5)

if 'date_end' in kwargs:
    date_end = str(kwargs['date_end'])
    end_dt = datetime.datetime.strptime(date_end + ' 23:59:55', '%Y%m%d %H:%M:%S')
else:
    end_dt = datetime.datetime(2020, 2, 5, 14, 59, 55)

if 'plot_range' in kwargs:
    plot_range = [0, int(kwargs['plot_range'])]
else:
    plot_range = [0, 3000]

#  read in moments
system = "LIMRAD94_30s"
radar_Z = larda.read(system, "Ze", [begin_dt, end_dt], plot_range)
radar_Z['var_unit'] = 'dBZ'
radar_Z['colormap'] = 'jet'
cloud_bases_tops = larda.read(system, "cloud_bases_tops", [begin_dt, end_dt], plot_range)
dt_list = np.asarray([datetime.datetime.utcfromtimestamp(time) for time in cloud_bases_tops['ts']])
range_list = cloud_bases_tops['rg']/1000.0  # convert m to km
var = cloud_bases_tops['var']
# mask -999 = fill value
var = np.ma.masked_where(var == -999, var)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_preliminary_{plot_range[1] / 1000:.0f}km_30s'

fig, ax = pyLARDA.Transformations.plot_timeheight(radar_Z, range_interval=plot_range, rg_converter=True, title=True,
                                                  z_converter='lin2z')
# create a binary color map for bases and tops
cmap = ListedColormap(["darkgreen", "darkmagenta"])
ax.pcolormesh(matplotlib.dates.date2num(dt_list[:]), range_list[:], np.transpose(var[:, :]),
              label='cloud bases and tops', cmap=cmap)
legend_elements = [Patch(facecolor='darkgreen', label='Bases'),
                   Patch(facecolor='darkmagenta', label='Tops')]
ax.legend(handles=legend_elements, title="Hydrometeor Layers")
fig.savefig(name + '_cbt_Z.png', dpi=250)
print(f'figure saved :: {name}_cbt_Z.png')

fig, ax = pyLARDA.Transformations.plot_timeheight(cloud_bases_tops, range_interval=plot_range, rg_converter=True,
                                                  title=True)
fig.savefig(name + '_cbt.png', dpi=250)
print(f'figure saved :: {name}_cbt.png')

