#!/usr/bin/env python
"""Script to add the cloud mask to the uploaded files on aeris in order to work on a combined lidar radar cloud mask
author: Johannes Roettenbacher
"""
import sys
LARDA_PATH = '/projekt1/remsens/work/jroettenbacher/Base/larda'
LARDA_PATH = 'C:/Users/Johannes/PycharmProjects/Base/larda'
sys.path.append(LARDA_PATH)
import xarray as xr
from functions_jr import find_bases_tops
import numpy as np
path = "./data"
file = "eurec4a_rv-meteor_cloudradar_20200211_v1.1.nc"

ds = xr.open_dataset(f"{path}/{file}")
mask = ~np.isnan(ds["Zh"])
ds["cloud_mask"] = mask
ds["cloud_mask"].attrs.update(dict(long_name="Cloud mask", units="1", comment="1: signal, 0: no signal"))
ds["cloud_mask"].attrs.pop("plot_range")
ds["cloud_mask"].attrs.pop("plot_scale")

ds.to_netcdf(f"{path}/{file.replace('.1.nc', '.2.nc')}", format="NETCDF4_CLASSIC")