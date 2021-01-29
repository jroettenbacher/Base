#!/usr/bin/env python
"""Compare different ways to calculate the heave rate at a specific point on a ship
1. Approach (Hannes Griesche):
    Calculate the cross product between the velocity vector of the ship and the position vector of the radar to
    transform the velocity vector of the ship to the radar position. Convert this new vector to the earth coordinate
    system and read out the z-value. Add on to that the heave motion of the ship itself.
2. Approach (Claudia Acquistapace):
    Transform the Position vector of the radar to the earth coordinate system and calculate the difference in the
    z-component for every time step.
"""

import pandas as pd
import functions_jr as jr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range('20200117', '20200228')  # define all dates of campaign
# read in data for 1. approach
seapath = pd.concat(jr.read_seapath(date) for date in dates)
seapath = jr.calc_heave_rate(seapath)
# read in data for 2. approach
seapath_ca = xr.concat((jr.read_seapath(date, output_format='xarray') for date in dates), dim='time')
seapath_ca = jr.f_shiftTimeDataset(seapath_ca)
seapath_ca = jr.calc_heave_rate_claudia(seapath_ca)
seapath_ca = seapath_ca.to_dataframe()

# calculate the difference in heave rate
hr_diff = pd.Series(seapath['heave_rate_radar'].values - seapath_ca['heave_rate_radar'].values, index=seapath.index)
# set plot style
jr.set_colorblind_friendly_plot_style()

fig, axs = plt.subplots(nrows=3)
axs[0].plot(seapath['heave_rate_radar'], label="Hannes' approach")
axs[1].plot(seapath_ca['heave_rate_radar'], label="Claudia's approach")
axs[2].plot(hr_diff, label="Hannes-Claudia")
for ax in axs:
    ax.legend()
    ax.grid()
    ax.set_xlabel("Date")
axs[1].set_ylabel("Heave Rate [m/s]")
axs[0].set_title("Difference between heave rate at radar\n"
                 " calculated by Hannes Griesche and Claudia Acquistapace")
ylim_up = axs[2].get_ylim()[1]
xlim_le = axs[2].get_xlim()[0] + 1
ypos = ylim_up - ylim_up / 2  # dynamically set y position of text box
axs[2].text(xlim_le, ypos, f"Mean: {hr_diff.mean():.2f}\nMedian: {hr_diff.median():.2f}\nStd: {hr_diff.std():.2f}",
            size=10, bbox=dict(fc='white'))
fig.autofmt_xdate()
fig.tight_layout()
plt.savefig("./tmp/heave_rate_diff_jr-ca.png")
plt.close()
