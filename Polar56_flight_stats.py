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
p5_infile = "/projekt_agmwend/home_rad/jroettenbacher/phd_base/polar5_nav_files.dat"
p6_infile = "/projekt_agmwend/home_rad/jroettenbacher/phd_base/polar6_nav_files.dat"
for infile, pattern in zip([p5_infile, p6_infile], ["**/horidata/Nav_files/**/*Polar5*.nav", "**/horidata/**/Data_GPS_P6*.dat"]):
    if not os.path.isfile(infile):
        files = glob.glob(os.path.join(root_path, pattern), recursive=True)
        outfile = open(infile, "w")
        for file in files:
            outfile.write(file)
            outfile.write("\n")
        outfile.close()
polar5_files = open(p5_infile).read().splitlines()
polar6_files = open(p6_infile).read().splitlines()
nautical_mile = 1852  # in meters
feet = 0.3048  # in meters

col_names = ["alt", "vel"]
df5 = pd.concat([pd.read_csv(file, sep="\s+", skiprows=3, names=col_names, usecols=[3, 4])
                 for file in polar5_files])
df6 = pd.concat([pd.read_csv(file, sep="\s+", skiprows=4, names=col_names, usecols=[8, 9])
                 for file in polar6_files])

for df, num in zip([df5, df6], [5, 6]):
    # convert from m to feet and m/s to nm/h
    df["alt_ft"] = df["alt"] / feet
    df["vel_nmh"] = df["vel"] * 3600 / nautical_mile
    # bin data and take average of each altitude bin
    bins = np.arange(0, 20000, 1000)
    df["bin"] = pd.cut(df["alt_ft"], bins)
    df_new = df.groupby(["bin"]).mean().loc[:, ["vel_nmh"]]

    # write a JSON formated Table for copy and paste
    file = open(f"/projekt_agmwend/home_rad/jroettenbacher/phd_base/p{num}_performance.JSON", "w")
    weight = np.repeat([0], 15)
    altitude = np.linspace(500, 14500, 15)
    fuel_consumption = np.repeat([3000], 15)
    for w, a, v, f in zip(weight, altitude, df_new['vel_nmh'][0:15].values.round(-1), fuel_consumption):
        file.write(f"[{w}, {a}, {v}, {f}],\n")

    file.close()
