#!/usr/bin/env python
"""Script to read in climate data for Limassol and Leipzig and plot a Climate chart
Source: Data downloaded from https://www.ecad.eu/dailydata/index.php
Note: used blended data
author: Johannes Röttenbacher
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

base_dir = "C:/Users/Johannes/PycharmProjects/Base"
# output dir
indir = f"{base_dir}/data/climate_data"
outdir = indir
# location = "leipzig_schkeuditz"  # can be cyprus or leipzig_schkeuditz
# location = "cyprus"  # can be cyprus or leipzig_schkeuditz
location = "weissenburg"  # can be cyprus or leipzig_schkeuditz
# select time frame for climatology
start_y, end_y = 1991, 2020
# toggle title
title = True
# %% read in data
filenames = glob.glob(f"{indir}/{location}/*STAID*.txt")
dfs = [pd.read_csv(file, skiprows=range(19), skipinitialspace=True,
                   na_values=-9999) for file in filenames]

# %% prepare data for plotting
dfs = [df.set_index(pd.to_datetime(df.DATE, format='%Y%m%d')).sort_index() for df in dfs]
dfs = [df.drop(['SOUID', 'DATE'], axis=1) for df in dfs]
df_all = dfs[0].join(dfs[1:], how='outer')

# convert the Temperature from 0.1°C to °C / Precipitation from 0.1mm to mm
for var in ['TG', 'TN', 'TX', 'RR']:
    df_all[var] = df_all[var] * 0.1

# resample to monthly means/sums for most recent climatological reference period 1991-2020
df_monthly = df_all.loc[f'{start_y}':f'{end_y}'].resample('M').agg(
    {'TG': 'mean', 'TN': 'mean', 'TX': 'mean', 'RR': 'sum'})
# add a month column
df_monthly['month'] = df_monthly.index.month
# get multiyear monthly mean
df_mymean = df_monthly.groupby('month').mean()
# add a column with the names of each month
df_mymean['Month'] = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dez']
# get yearly average
yearly_sum = df_mymean['RR'].sum()

# %% plot climate chart
font = {'size': 12, 'sans-serif': ['Times New Roman']}
figure = {'figsize': [10, 12], 'dpi': 300}  # A4 filling
plt.rc('font', **font)
plt.rc('figure', **figure)
# set new colorblind friendly color cycle
CB_color_cycle = ["#6699CC", "#CC6677", "#117733", "#DDCC77"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CB_color_cycle)

# %% actual plotting
fig, ax1 = plt.subplots(figsize=(6.3, 3.3))
ax2 = ax1.twinx()
ax1.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
ax1.patch.set_visible(False)  # hide the 'canvas'
line1, = ax1.plot(df_mymean['Month'], df_mymean['TN'], label='Minimum Temperature')
line2, = ax1.plot(df_mymean['Month'], df_mymean['TX'], label='Maximum Temperature')
line3, = ax1.plot(df_mymean['Month'], df_mymean['TG'], label='Mean Temperature')
bar = ax2.bar(df_mymean['Month'], df_mymean['RR'], color='#0868ac', label="Precipitation")
ax1.set_xticks(np.arange(len(df_mymean.Month)))
ax1.set_xticklabels(df_mymean.Month)
ax1.set_ylim([-5, 35])
ax1.grid()
ax1.set_ylabel("Temperature [°C]")
ax1.set_xlabel("Month")
ax2.set_ylabel("Precipitation [mm]")
ax2.set_ylim([0, 120])
legend_elements = [line1, line2, line3, bar]
ax1.legend(handles=legend_elements, bbox_to_anchor=(1, -0.2), ncol=2)
if title:
    ax1.set_title(f"{location.capitalize()} {start_y} - {end_y}")
# add yearly sum in text box
ax1.text(0.02, 0.85, f"Annual mean: {yearly_sum:.1f} mm", transform=ax1.transAxes, fontsize=10,
         bbox=dict(ec="black", fc="white"))
plt.tight_layout()
plt.subplots_adjust(bottom=0.32)
figname = f"{outdir}/{location}_climate_chart.png"
plt.savefig(figname, dpi=300)
print(f"Saved {figname}")
# plt.show()
plt.close()
