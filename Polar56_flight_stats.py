#!/usr/bin/env python
"""File to search for navigation data files from the Polar 5 and 6 aircraft and create statistics about flight speed
and altitude

author: Johannes Roettenbacher
"""

import glob
import os
import pandas as pd
import numpy as np

root_path = "/projekt_agmwend/data_raw"
polar5_files = glob.glob(os.path.join(root_path, "**/*Polar5*.nav"), recursive=True)
polar6_files = glob.glob(os.path.join(root_path, "**/*Polar6*.nav"), recursive=True)
nautical_mile = 1852  # in meters
feet = 0.3048  # in meters

file = polar5_files[0]
column_names = ["time", "lon", "lat", "alt", "vel", "pitch", "roll", "yaw", "sza", "saa"]
data = pd.concat([pd.read_csv(file, sep="\s+", skiprows=3, names=column_names, usecols=[3, 4]) for file in polar5_files])
df = data.loc[:, ["alt", "vel"]]
# convert from m to feet and m/s to nm/h
df["alt_ft"] = df["alt"] / feet
df["vel_nmh"] = df["vel"] * 3600 / nautical_mile
# bin data and take average of each altitude bin
bins = np.arange(0, 20000, 1000)
df["bin"] = pd.cut(df["alt_ft"], bins)
df_new = df.groupby(["bin"]).mean().loc[:, ["vel_nmh"]]

