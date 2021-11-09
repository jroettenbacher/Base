#!/usr/bin/env python
"""Script to add the cloud mask to the uploaded files on aeris in order to work on a combined lidar radar cloud mask
author: Johannes Roettenbacher
"""
import sys
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
LARDA_PATH = '/projekt1/remsens/work/jroettenbacher/Base/larda'
sys.path.append(LARDA_PATH)
path = Path("/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/limrad94/upload_to_aeris_v1.1")
filepaths = path.glob('*.nc')

for filepath in tqdm(filepaths):
    ds = xr.open_dataset(filepath)
    mask = ~np.isnan(ds["Zh"])
    ds["hydrometeor_mask"] = mask
    ds["hydrometeor_mask"].attrs.update(dict(long_name="Hydrometeor mask", units="1", comment="1: signal, 0: no signal"))
    ds["hydrometeor_mask"].attrs.pop("plot_range")
    ds["hydrometeor_mask"].attrs.pop("plot_scale")
    ds.attrs["version"] = ds.attrs["version"] + ", 1.2: added hydrometeor_mask"

    outpath = Path(str(filepath).replace("1.1", "1.2"))
    ds.to_netcdf(outpath, format="NETCDF4_CLASSIC")
