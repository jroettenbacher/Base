#!/usr/bin/env python
"""Plot RV Meteor DSHIP data
author: Johannes Roettenbacher
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# local path
# indir = "C:/Users/Johannes/Documents/EUREC4A/data/dship"
file = "RV-Meteor_lat_lon_1Hz.dat"
# server path
indir = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP/upload_to_aeris"
files = [f for f in os.listdir(indir) if "1Hz" in f]

df = pd.read_csv(f"{indir}/{file}")
df.columns = ["datetime", "lon", "lat"]
df['datetime'] = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S")
df = df.set_index('datetime')

df.lat.loc["20200117":"20200219"].plot()
plt.grid()
plt.title("RV-Meteor latitude from DSHIP 1Hz")
plt.show()
plt.close()

df.lon.loc["20200117":"20200219"].plot()
plt.grid()
plt.title("RV-Meteor longitude from DSHIP 1Hz")
plt.show()
plt.close()