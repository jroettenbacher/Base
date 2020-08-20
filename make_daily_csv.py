#!/usr/bin/env python
"""Make daily csv files from the sniffer file from Johannes Buehl
input: cloud_collection_LEIPZIG_all.csv
output: .../sniffer/leipzig/YYYY/YYYY_mm_dd_cloud_collection_LEIPZIG_all.csv
"""

import pandas as pd
import os

path = "C:/Users/Johannes/PycharmProjects/cirrus_dynamics/data/sniffer/leipzig"
infile = "cloud_collection_LEIPZIG_all.csv"
df = pd.read_csv(f'{path}/{infile}', sep=';')
df.Begin_Date = pd.to_datetime(df.Begin_Date, format='%Y_%m_%d_%H_%M_%S')

for date in df.Begin_Date.dt.date.unique():
    outpath = f'{path}/{date.year}'
    # create output folder if it does not exist yet
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    tmp_df = df.loc[df.Begin_Date.dt.date == date]  # select only rows which match the current date
    tmp_df.to_csv(f'{outpath}/{date:%Y_%m_%d}_cloud_collection_LEIPZIG_all.csv', sep=';')  # save to csv



