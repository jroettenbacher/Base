#!/bin/python

"""Script to plot single spectra before and after heave correction
"""

########################################################################################################################
# library import
########################################################################################################################
import sys
import datetime as dt
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
from pyLARDA.SpectraProcessing import heave_correction, heave_correction_spectra
import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

plot_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction_single_spectra"

# connect to larda
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

# define test cases
begin_dt = dt.datetime(2020, 2, 10, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 10, 23, 59, 55)
begin_dt_zoom = dt.datetime(2020, 2, 10, 22, 10, 0)
end_dt_zoom = dt.datetime(2020, 2, 10, 22, 20, 0)
# define time and height of spectra
time = dt.datetime(2020, 2, 10, 22, 16, 0)
plot_range = 2000

data = dict()
data['VHSpec'] = larda.read('LIMRAD94', 'VSpec', [begin_dt_zoom, end_dt_zoom], [plot_range])
data['n_ts'], data['n_rg'], data['n_vel'] = data['VHSpec']['var'].shape
for var in ['C1Range', 'C2Range', 'C3Range', 'SeqIntTime', 'Inc_ElA', 'MaxVel', 'DoppLen']:
    print('loading variable from LV1 :: ' + var)
    data.update({var: larda.read('LIMRAD94', var, [begin_dt, end_dt], [0, 'max'])})
data['SeqIntTime'] = data['SeqIntTime']['var'][0]
data['MaxVel'] = data['MaxVel']['var'][0]
data['DoppLen'] = data['DoppLen']['var'][0]
data['VHSpec_shift'] = data['VHSpec']
data['VHSpec_shift']['name'] = 'VHSpec_shifted'

new_spec, heave_cor_array_2, n_dopp_bins_shift, seapath_out_2 = heave_correction_spectra(data, begin_dt)

data['VHSpec_shift'] = h.put_in_container(new_spec, data['VHSpec_shift'])
spectrum1 = larda.read('LIMRAD94', 'VSpec', [time], [plot_range])
spectrum2 = data['VHSpec_shift']
name = "spectra"
fig, ax = pyLARDA.Transformations.plot_spectra(spectrum1, spectrum2, z_converter='lin2z', vmin=-50, vmax=20, title=True)
fig.savefig(f'{plot_path}/{name}_{time.strftime("%Y%m%d_%H%M%S")}_{plot_range}m.png', dpi=150)

