#!/usr/bin/env python
"""script to analyse virgae during eurec4a
Idea: compare height of first radar echo and cloud base height detected by ceilometer
Step1: find virga in each time step
Step2: create virga mask
Step3: detect single virga and define borders
Step4: evaluate statistcally -> output like cloud sniffer in csv file
Result:
    - depth of virga
    - maximum Ze in virga
    - dataset
"""

import sys
sys.path.append("/projekt1/remsens/work/jroettenbacher/Base/larda")
import pyLARDA
import pyLARDA.helpers as h
import functions_jr as jr
import datetime as dt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.dates import date2num
from matplotlib.patches import Polygon
import pandas as pd
import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# gather command line arguments
method_name, args, kwargs = h._method_info_from_argv(sys.argv)

save_fig = False  # plot the two virga masks? saves to ./tmp/
save_csv = True
plot_data = True  # plot radar Ze together with virga polygons
csv_outpath = '/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/virga_sniffer'
larda = pyLARDA.LARDA().connect("eurec4a")

if 'date' in kwargs:
    date = str(kwargs['date'])
    begin_dt = dt.datetime.strptime(date, "%Y%m%d")
else:
    begin_dt = dt.datetime(2020, 1, 20, 0, 0, 0)
end_dt = begin_dt + dt.timedelta(hours=23, minutes=59, seconds=59)
time_interval = [begin_dt, end_dt]

# read in data
radar_ze = larda.read("LIMRAD94_cn_input", "Ze", time_interval, [0, 'max'])
ceilo_cbh = larda.read("CEILO", "cbh", time_interval)
rainrate = jr.read_rainrate()  # read in rain rate from RV-Meteor DWD rain sensor
rainrate = rainrate.sort_index()[time_interval[0]:time_interval[1]]  # sort index and select time interval

# make a rain flag, extend rain flag x minutes after last rain to account for wet radome
rain_flag_dwd = rainrate.Dauer > 0  # set rain flag if rain duration is greater 0 seconds
# get a one dimensional array with the indices where rainflag turns from True to False or vice versa
indices = np.asarray(np.where(np.diff(rain_flag_dwd))).flatten()
# get indices where rainflag turns from True to False only -> index where it stops to rain
rain_indices = np.asarray([idx for idx in indices if rain_flag_dwd[idx]])
# from the end of each rain event add 10 minutes of masked values
minutes = 10  # just for readability
for i in rain_indices:
    rain_flag_dwd[i:(i+minutes)] = True

# interpolate radar and rain rate on ceilo time
radar_ze_ip = pyLARDA.Transformations.interpolate2d(radar_ze, new_time=ceilo_cbh['ts'])
radar_ze_ip['mask'] = radar_ze_ip['mask'] == 1  # turn mask from integer to bool

f_rr = interp1d(h.dt_to_ts(rain_flag_dwd.index), rain_flag_dwd, kind='nearest', fill_value="extrapolate")
rain_flag_dwd_ip = f_rr(ceilo_cbh['ts'])  # interpolate DWD RR to ceilo time values
rain_flag_dwd_ip = rain_flag_dwd_ip == 1  # turn mask from integer to bool

# get height of first ceilo cloud base and radar echo
h_ceilo = ceilo_cbh['var'].data[:, 0]
# get arrays which have the indices at which a signal is measured in a timestep, each array corresponds to one timestep
rg_radar_all = [np.asarray(~radar_ze_ip['mask'][t, :]).nonzero()[0] for t in range(radar_ze_ip['ts'].shape[0])]

# loop through arrays and select first element which corresponds to the first range gate with a signal
# convert the range gate index into its corresponding height
# if the time stamp has no signal an empty array is returned, append a -1 for those steps to keep size of time dimension
h_radar = list()
for i in rg_radar_all:
    try:
        h_radar.append(radar_ze_ip['rg'][i[0]])
    except IndexError:
        h_radar.append(-1)

########################################################################################################################
# Step 1: Is there a virga in the time step
########################################################################################################################
h_radar = np.asarray(h_radar)  # convert list to numpy array
cloudy = h_radar != -1  # does the radar see a cloud?
# since both instruments have different range resolutions compare their heights and decide if their are equal within a
# tolerance of 23m (approximate range resolution of first radar chirp)
h_diff = ~np.isclose(h_ceilo, h_radar, atol=23)  # is the ceilometer cloud base different from the first radar echo height?
virga = h_ceilo > h_radar  # is the ceilometer cloud base higher than the first radar echo?
# combine all masks
virga = cloudy & h_diff & virga & ~rain_flag_dwd_ip  # is a virga present in the time step?, exclude rainy profiles

########################################################################################################################
# Step 2: Create Virga Mask
########################################################################################################################
# virga mask on ceilo resolution
# if timestep has virga, mask all radar range gates between first radar echo and cbh from ceilo as virga
# find equivalent range gate to ceilo cbh
virga_mask = np.zeros_like(radar_ze_ip['var'])
for i in np.where(virga)[0]:
    lower_rg = rg_radar_all[i][0]
    upper_rg = h.argnearest(radar_ze_ip['rg'], h_ceilo[i])
    assert lower_rg < upper_rg, f"Lower range ({lower_rg}) higher than upper range ({upper_rg})"
    virga_mask[i, lower_rg:upper_rg] = 1

virga_mask = virga_mask == 1
# make a larda container with the mask
virga = h.put_in_container(virga_mask, radar_ze_ip, name="virga_mask", paramkey="virga", var_unit="-", var_lims=[0, 1])
location = virga['paraminfo']['location']
if save_fig:
    fig, ax = pyLARDA.Transformations.plot_timeheight2(virga, range_interval=[0, 1000])
    virga_dt = [h.ts_to_dt(t) for t in virga['ts']]
    ax.plot(virga_dt, h_ceilo, ".", color="purple")
    figname = f"./tmp/{location}_virga_ceilo-res_{time_interval[0]:%Y%m%d}.png"
    fig.savefig(figname)
    log.info(f"Saved {figname}")

# virga mask on radar resolution

ts_list = list()
for t in ceilo_cbh['ts']:
    id_diff_min = h.argnearest(radar_ze['ts'], t)  # find index of nearest radar time step to ceilo time step
    ts_list.append(id_diff_min)  # append index to list

virga_mask_hr = np.zeros_like(radar_ze['mask'])
for j in range(len(ts_list)-1):
    ts1 = ts_list[j]
    ts2 = ts_list[j+1]
    if any(virga_mask[j]) and any(virga_mask[j+1]):
        rg1 = np.where(virga_mask[j])[0][0]  # select first masked range gate
        rg2 = np.where(virga_mask[j+1])[0][-1]  # select last masked range gate
        virga_mask_hr[ts1:ts2, rg1:rg2] = True  # interpolate mask to radar time resolution

virga_hr = h.put_in_container(virga_mask_hr, radar_ze, name="virga_mask", paramkey="virga", var_unit="-",
                              var_lims=[0, 1])
if save_fig:
    fig, ax = pyLARDA.Transformations.plot_timeheight2(virga_hr, range_interval=[0, 1000])
    virga_dt = [h.ts_to_dt(t) for t in virga['ts']]
    ax.plot(virga_dt, h_ceilo, ".", color="purple")
    figname = f"./tmp/{location}_virga_radar-res_{time_interval[0]:%Y%m%d}.png"
    fig.savefig(figname)
    log.info(f"Saved {figname}")

########################################################################################################################
# Step 3: define single virga borders (corners)
########################################################################################################################
min_vert_ext = 3  # minmum verticalextent: 3 radar range gates (70 - 120m) depending on chirp
min_hori_ext = 20  # virga needs to be present for at least 1 minute (20+3s) to be counted
max_hori_gap = 10  # maximum horizontal gap: 10 radar time steps (40 - 30s) depending on chirp table
virgae = dict(ID=list(), virga_thickness_avg=list(), virga_thickness_med=list(), virga_thickness_std=list(),
              max_Ze=list(), min_Ze=list(), avg_height=list(), max_height=list(), min_height=list(),
              idx=list(), borders=list(), points_b=list(), points_t=list())
t_idx = 0
while t_idx < len(virga_hr['ts']):
    # check if a virga was detected in this time step
    if any(virga_hr['var'][t_idx, :]):
        v, b, p_b, p_t = list(), list(), list(), list()
        # as long as a virga is detected within in the maximum horizontal gap add the borders to v
        while virga_hr['var'][t_idx:(t_idx+max_hori_gap), :].any():
            h_ids = np.where(virga_hr['var'][t_idx, :])[0]
            if len(h_ids) > 0:
                if (h_ids[-1] - h_ids[0]) > min_vert_ext:
                    v.append((t_idx, h_ids[0], h_ids[-1]))
                    b.append((virga_hr['ts'][t_idx], virga_hr['rg'][h_ids[0]], virga_hr['rg'][h_ids[-1]]))
                    p_b.append(np.array([date2num(h.ts_to_dt(virga_hr['ts'][t_idx])), virga_hr['rg'][h_ids[0]]]))
                    p_t.append(np.array([date2num(h.ts_to_dt(virga_hr['ts'][t_idx])), virga_hr['rg'][h_ids[-1]]]))
            t_idx += 1
        # when the virga is finished add the list of borders to the output list
        if len(v) > min_hori_ext:
            virgae['idx'].append(v)
            virgae['borders'].append(b)
            virgae['points_b'].append(p_b)
            virgae['points_t'].append(p_t)
    else:
        t_idx += 1

########################################################################################################################
# Step 4: get statistics of each virga and save to csv file
########################################################################################################################
# loop through virgas, select radar pixels, get stats
for v in virgae['idx']:
    time_slice = [v[0][0], v[-1][0]]
    # get only range borders
    rgs = [k[1:] for k in v]
    range_slice = [np.min(rgs), np.max(rgs)]
    virga_ze = pyLARDA.Transformations.slice_container(radar_ze, index={'time': time_slice, 'range': range_slice})
    mask = pyLARDA.Transformations.slice_container(virga_hr, index={'time': time_slice, 'range': range_slice})['var']
    # calculate thickness in each timestep
    thickness = list()
    for idx in range(len(v)):
        rg = radar_ze['rg']
        thickness.append(rg[v[idx][2]] - rg[v[idx][1]])
    # add stats do dictionary
    virgae['max_Ze'].append(h.lin2z(np.max(virga_ze['var'][mask])))
    virgae['min_Ze'].append(h.lin2z(np.min(virga_ze['var'][mask])))
    virgae['avg_height'].append(np.mean(virga_ze['rg']))
    virgae['max_height'].append(np.max(virga_ze['rg']))
    virgae['min_height'].append(np.min(virga_ze['rg']))
    virgae['virga_thickness_avg'].append(np.mean(thickness))
    virgae['virga_thickness_med'].append(np.median(thickness))
    virgae['virga_thickness_std'].append(np.std(thickness))
    virgae['ID'].append(dt.datetime.strftime(h.ts_to_dt(virga_ze['ts'][0]), "%Y%m%d%H%M%S%f"))

if save_csv:
    # write to csv file
    csv_out = pd.DataFrame(virgae)
    csv_name = f"{csv_outpath}/{location}_virga-collection_{time_interval[0]:%Y%m%d}.csv"
    csv_out.to_csv(csv_name, sep=';', index=False)
    log.info(f"Saved {csv_name}")


########################################################################################################################
# Step 5: plot radar Ze with virga in boxes (optionally)
########################################################################################################################
if plot_data:
    radar_ze.update(var_unit="dBZ", var_lims=[-60, 20])
    t = [begin_dt, begin_dt + dt.timedelta(hours=12), end_dt]
    for i in range(2):
        fig, ax = pyLARDA.Transformations.plot_timeheight2(radar_ze, range_interval=[0, 2000],
                                                           time_interval=[t[i], t[i+1]], z_converter='lin2z')
        for points_b, points_t in zip(virgae['points_b'], virgae['points_t']):
            # append the top points to the bottom points in reverse order for drawing a polygon
            points = points_b + points_t[::-1]
            ax.add_patch(Polygon(points, closed=True, fc='pink', ec='purple', alpha=0.7))
        figname = f"{csv_outpath}/{location}_radar-Ze_virga-masked_{time_interval[0]:%Y%m%d}_{i+1}.png"
        fig.savefig(figname)
        log.info(f"Saved {figname}")
