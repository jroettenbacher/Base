
import datetime
import sys
# path to local larda source code - no data needed locally
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import numpy as np
import time
import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.Transformations as pLTransf
import matplotlib.pyplot as plt

# optionally setup logging
import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

begin_dt = datetime.datetime(2020, 1, 25, 0, 0, 5)
end_dt = datetime.datetime(2020, 1, 26, 23, 59, 30)
plot_range = 3000

plot_path = "/projekt1/remsens/work/jroettenbacher/Base/plots"

larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
t1 = time.time()

Ze_LIMRAD = larda.read("LIMRAD94_cn_input", "Ze", [begin_dt, end_dt], [0, 'max'])
Ze_LIMRAD['var_unit'] = 'dBZ'
cbh = larda.read("CEILO", "cbh", [begin_dt, end_dt])

time_list = cbh['ts']
var = cbh['var'].copy()
var = np.ma.masked_where(var < 100, var)

# var = np.ma.masked_values(var, -1)
dt_list = np.asarray([datetime.datetime.utcfromtimestamp(time) for time in time_list])

fig, ax = pyLARDA.Transformations.plot_timeheight(Ze_LIMRAD, z_converter='lin2z', range_interval=[0, plot_range],
                                                  title=True)
ax.plot(dt_list, var, '.', ms=2.5, label='cloud base height ceilometer', color='purple', alpha=0.7)
# ax.plot(dt_list, var[:, 1], 's', ms=3, label='cloud base height 2', color='green', alpha=0.7)
# ax.plot(dt_list, var[:, 2], 'p', ms=3, label='cloud base height 3', color='blue', alpha=0.7)
# ax.legend()
fig.savefig(f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range / 1000:.0f}km_Ze_ceilo_cbh.png')
print(f"saved figure to {plot_path}\nTime needed {time.time() - t1}")
