#!/bin/usr/env python
import sys, datetime, time

sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import pyLARDA
from larda.pyLARDA.SpectraProcessing import load_spectra_rpgfmcw94
import matplotlib.pyplot as plt
import numpy as np

larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
# begin_dt = datetime.datetime(2020, 2, 3, 0, 0, 5)
# end_dt = datetime.datetime(2020, 2, 3, 23, 59, 55)
# begin_dt_zoom = datetime.datetime(2020, 2, 3, 18, 0, 0)
# end_dt_zoom = datetime.datetime(2020, 2, 3, 18, 20, 0)
begin_dt = datetime.datetime(2020, 2, 5, 0, 0, 5)
end_dt = datetime.datetime(2020, 2, 5, 23, 59, 55)
begin_dt_zoom = datetime.datetime(2020, 2, 5, 9, 25, 0)
end_dt_zoom = datetime.datetime(2020, 2, 5, 9, 35, 0)
timespan = [begin_dt, end_dt]
timespan_zoom = [begin_dt_zoom, end_dt_zoom]
container = load_spectra_rpgfmcw94(larda, timespan_zoom, heave_correction=True)
container_noheave = load_spectra_rpgfmcw94(larda, timespan_zoom)

spectrum = container['VHSpec'].copy()
spectrum['var'] = np.ma.masked_where(spectrum['var'][100, 50, :] == -999, spectrum['var'][100, 50, :])
plt.plot(spectrum['var'])
plt.savefig(f'../plots/spectra_heave_cor.png')
plt.close()

spectrum = container_noheave['VHSpec'].copy()
spectrum['var'] = np.ma.masked_where(spectrum['var'][100, 50, :] == -999, spectrum['var'][100, 50, :])
plt.plot(spectrum['var'])
plt.savefig(f'../plots/spectra_no_heave_cor.png')
plt.close()
