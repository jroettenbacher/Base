#!/usr/bin/python
"""Download selected ERA5 data from the Climate Data Store
author: Johannes Roettenbacher
Note: ERA5 data is stored so that each month is on one tape. Thus it should be faster to retrieve single months.
=> loop over years and months to get data faster
"""

import cdsapi
from functions_jr import days_of_month
import os
import time
c = cdsapi.Client()
years = range(2010, 2017)  # set years to retrieve
months = range(1, 13)  # set months to retrieve
times = [f'{hour:02}:00' for hour in range(0, 24)]  # set hours to retrieve
area = [51.5, 12.25, 51.25, 12.5]  # set area, N W S E
p_levels = [str(z) for z in ([1, 50] + list(range(100, 300, 25))
                             + list(range(300, 750, 50))
                             + list(range(750, 1025, 25)))]  # set pressure levels to retrieve

start = time.time()
print("#################################\nStart downloading ERA5 data\n#################################")
for year in years:
    t1 = time.time()
    print(f"####\n# Downloading year {year} #\n####")
    path = f"/poorgafile2/remsens/data/era5/leipzig/{year}"  # set output directory
    # create output directory if it doesn't exist yet
    if not os.path.isdir(path):
        os.mkdir(path)
    for month in months:
        t2 = time.time()
        dates = days_of_month(year, month)
        filename = f"{path}/era5_{year}_{month}_pl.nc"
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['fraction_of_cloud_cover', 'relative_humidity', 'specific_cloud_ice_water_content',
                             'specific_cloud_liquid_water_content', 'specific_humidity', 'temperature',
                             'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity'
                             ],
                'pressure_level': p_levels,
                'date': dates,
                'time': times,
                'area': area,
            },
            f'{filename}')
        print(f"Downloaded month {month} in {time.time() - t2:.2f} seconds")
    print(f"Done with year {year} in {time.time() - t1:.2f} seconds")
print("##################\nDone with ERA5 download\n##################")
