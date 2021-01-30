#!/usr/bin/env python
"""script to create csv files with datetime, lat, lon for EUREC4A cruise of Meteor
can be used by Claudia Acquistapace to retrieve ICON grid cells"""

import functions_jr as jr
import datetime as dt
import pandas as pd

path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"
# make date range for whole campaign
start_date = dt.date(2020, 1, 17)
end_date = dt.date(2020, 2, 28)
dates = pd.date_range(start_date, end_date)

df_out = pd.DataFrame()
for date in dates:
    d = date.strftime("%Y%m%d")
    df = jr.read_dship(d)[['SYS.STR.PosLon', 'SYS.STR.PosLat']]
    df_out = df_out.append(df)

# write output csv files
df_out.to_csv(f"{path}/RV-Meteor_lat_lon_1Hz.dat", sep=',')