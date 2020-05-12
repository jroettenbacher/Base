#!/usr/bin/python

"""Merge daily nc2 files into monthly nc4 files"""

import os
from cdo import *
cdo = Cdo()

wkdir = "/poorgafile2/remsens/data/era5/leipzig"
if not os.getcwd() == wkdir:
    os.chdir(wkdir)

years = sorted(os.listdir())
months = range(13)

for year in years:
    os.chdir(f"{wkdir}/{year}")
    for month in months:
        cdo.mergetime(input=f"era5_pl_{year}-{month:02}-*.nc", output=f"era5_pl_{year}-{month:02}.nc",
                      options="-f nc4 -C -O -r")
