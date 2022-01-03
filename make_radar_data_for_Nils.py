#!/usr/bin/env python
"""Select only a few radar variables for Nils Walpers BA to reduce file size and ease of handling

author: Johannes Röttenbacher
"""

import os
import xarray as xr
from tqdm import tqdm

base_path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments"
inpath = f"{base_path}/limrad94/upload_to_aeris_v1.2"
inpath_ceilo = f"{base_path}/RV-METEOR_CEILOMETER/upload_to_aeris_v1"
outpath = f"{base_path}/limrad94/data_for_Nils"

files = sorted([f for f in os.listdir(inpath) if f.endswith(".nc")])
ceilo_files = sorted([f for f in os.listdir(inpath_ceilo) if f.endswith(".nc")])
# remove the files which do not have a corresponding radar file
ceilo_files.pop(10)
ceilo_files.pop(11)
ceilo_files.pop(11)
ceilo_files.pop(11)
variables = ["Zh", "hydrometeor_mask", "cloud_bases_tops", "latitude", "longitude", "lwp", "v"]

for i in tqdm(range(len(files))):
    ds = xr.open_dataset(f"{inpath}/{files[i]}")
    ceilo = xr.open_dataset(f"{inpath_ceilo}/{ceilo_files[i]}")
    ds = ds[variables]
    ds = ds.interp(time=ceilo.time)
    ds.to_netcdf(f"{outpath}/{files[i].replace('.nc', '_sel.nc')}")
