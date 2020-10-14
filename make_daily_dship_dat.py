#!usr/bin/env/ python
"""script to create daily dat files from 1Hz DSHIP data from RV-Meteor
author: Johannes Roettenbacher"""

import dask.dataframe as dd

path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"
filename = "DSHIP_all_1Hz.dat"

ddf = dd.read_csv(f"{path}/{filename}", skiprows=[1, 2], sep='\t',
                  dtype={'FOG.HEHDT.heading': 'float64',
                         'FOG.PPPRP.pitch': 'float64',
                         'FOG.PPPRP.roll': 'float64',
                         'SEAPATH.PSXN.Heading': 'float64',
                         'SEAPATH.PSXN.Heave': 'float64',
                         'SEAPATH.PSXN.Pitch': 'float64',
                         'SEAPATH.PSXN.Roll': 'float64',
                         'SYS.CALC.SPEED_kmh': 'float64',
                         'SYS.STR.HDG': 'float64',
                         'SYS.STR.PosLat': 'float64',
                         'SYS.STR.PosLon': 'float64',
                         'SYS.STR.Speed': 'float64',
                         'WEATHER.PBWWI.AirPress': 'float64',
                         'WEATHER.PBWWI.AirTempPort': 'float64',
                         'WEATHER.PBWWI.AirTempStarboard': 'float64',
                         'WEATHER.PBWWI.DewPointPort': 'float64',
                         'WEATHER.PBWWI.DewPointStarboard': 'float64',
                         'WEATHER.PBWWI.GlobalRadiation': 'float64',
                         'WEATHER.PBWWI.HumidityPort': 'float64',
                         'WEATHER.PBWWI.HumidityStarboard': 'float64',
                         'WEATHER.PBWWI.LwRadiation': 'float64',
                         'WEATHER.PBWWI.PyrradiometerTemp': 'float64',
                         'WEATHER.PBWWI.RealWindDir': 'float64',
                         'WEATHER.PBWWI.RelWindSpeed': 'float64',
                         'WEATHER.PBWWI.TrueWindDir': 'float64',
                         'WEATHER.PBWWI.TrueWindSpeed': 'float64',
                         'WEATHER.PBWWI.WaterTempPort': 'float64',
                         'WEATHER.PBWWI.WaterTempStarboard': 'float64',
                         'WEATHER.PBWWI.Visibility': 'float64'}
                  )

ddf['date time'] = dd.to_datetime(ddf['date time'])
ddf.head()
for date in ddf['date time'].dt.date.unique().compute():
    filename = f"{path}/RV-Meteor_DSHIP_all_1Hz_{date:%Y%m%d}.csv"
    tmp_df = ddf.loc[ddf['date time'].dt.date == date]  # select only rows which match the current date
    tmp_df.to_csv(filename, index=False, single_file=True)  # save to csv
