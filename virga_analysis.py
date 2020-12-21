#!/usr/bin/env python
"""script to analyse virgae during eurec4a
Idea: compare height of first radar echo and cloud base height detected by ceilometer
Step1: find virga in each time step
Step2: create virga mask
Step3: detect single virga and evaluate statistcally -> output like cloud sniffer in csv file
Result:
    - depth of virga
    - maximum Ze in virga
    - dataset
"""

import sys
sys.path.append("/projekt1/remsens/work/jroettenbacher/Base/larda")
import pyLARDA
import pyLARDA.helpers as h
import datetime as dt
import numpy as np
import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

larda = pyLARDA.LARDA().connect("eurec4a")

# define first and second part of campaign with differing chirp tables
time_interval1 = [dt.datetime(2020, 1, 20, 0, 0, 5), dt.datetime(2020, 1, 20, 23, 59, 55)] #  dt.datetime(2020, 1, 27, 23, 59, 55)]
time_interval2 = [dt.datetime(2020, 2, 1, 0, 0, 5), dt.datetime(2020, 2, 27, 23, 59, 55)]

# read in data
radar_ze = larda.read("LIMRAD94_cn_input", "Ze", time_interval1, [0, 'max'])
ceilo_cbh = larda.read("CEILO", "cbh", time_interval1)
ceilo_beta = larda.read("CEILO", "beta", time_interval1, [0, 5000])
# interpolate radar on ceilo time
radar_ze = pyLARDA.Transformations.interpolate2d(radar_ze, new_time=ceilo_cbh['ts'])
# turn mask from integer to bool
radar_ze['mask'] = radar_ze['mask'] == 1

# get height of first ceilo cloud base and radar echo
h_ceilo = ceilo_cbh['var'].data[:, 0]
# get arrays which have the indices at which a signal is measured in a timestep, each array corresponds to one timestep
rg_radar_all = [np.asarray(~radar_ze['mask'][t, :]).nonzero()[0] for t in range(radar_ze['ts'].shape[0])]

# loop through arrays and select first element which corresponds to the first range gate with a signal
# convert the range gate index into its corresponding height
# if the time stamp has no signal an empty array is returned, append a -1 for those steps to keep size of time dimension
h_radar = list()
for i in rg_radar_all:
    try:
        h_radar.append(radar_ze['rg'][i[0]])
    except IndexError:
        h_radar.append(-1)

########################################################################################################################
# Step 1: Is there a virga in the time step
########################################################################################################################
h_radar = np.asarray(h_radar)  # convert list to numpy array
# since both instruments have different range resolutions compare their heights and decide if their are equal within a
# tolerance of 23m (approximate range resolution of first radar chirp)
cloudy = h_radar != -1  # does the radar see a cloud?
h_diff = ~np.isclose(h_ceilo, h_radar, atol=23)  # is the ceilometer cloud base different from the first radar echo height?
virga = h_ceilo > h_radar  # is the ceilometer cloud base higher than the first radar echo?
# combine both masks
virga = cloudy & h_diff & virga  # is a virga present in the time step?

########################################################################################################################
# Step 2: Create Virga Mask
########################################################################################################################
# virga mask on ceilo resolution
# if timestep has virga, mask all radar range gates between first radar echo and cbh from ceilo as virga
# find equivalent range gate to ceilo cbh
virga_mask = np.zeros_like(radar_ze['var'])
for i in np.where(virga)[0]:
    lower_rg = rg_radar_all[i][0]
    upper_rg = h.argnearest(radar_ze['rg'], h_ceilo[i])
    assert lower_rg < upper_rg, f"Lower range ({lower_rg}) higher than upper range ({upper_rg})"
    virga_mask[i, lower_rg:upper_rg] = 1

virga_mask = virga_mask == 1
# make a larda container with the mask
virga = h.put_in_container(virga_mask, radar_ze, name="virga_mask", paramkey="virga", var_unit="-", var_lims=[0, 1])
fig, ax = pyLARDA.Transformations.plot_timeheight2(virga, range_interval=[0, 1000])
virga_dt = [h.ts_to_dt(t) for t in virga['ts']]
ax.plot(virga_dt, h_ceilo, ".", color="purple")
fig.savefig("./tmp/virga_ceilo-res.png")
# virga mask on radar resolution

