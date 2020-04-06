#!/usr/bin/env python
"""Skript to stich together all files from 27.1 - 31.1 of eurec4a
Lots of settings were changed in those four days so larda cannot handle those files
This skript reads in all files and stiches them together into daily cloudnet input files
Dates and Settings:
27.1 00:14-02:37 UTC no data, too huge files, data gap
27.1 02:37-03:03 UTC no data, file save bug, data gap
27.1 03:07-13:03 UTC TRADEWIND_CU_UNLIMITED_UNCOMPRESSED.MDF NF6
27.1 13:03-14:09 UTC 1SEC_NOISEFAC1.MDF NF1 -> probably no data because of LV0 file size, data gap
27.1 14:09-15:08 UTC 1SEC_UNCOMPRESSED.MDF NF0 -> probably no data because of LV0 file size, data gap
27.1 15:08-21:03 UTC TRADEWIND_CU_UNLIMITED_UNCOMPRESSED.MDF NF6
27.1 21:03-22:01 UTC 1SEC_NOISEFAC6.MDF NF6
27.1 22:01-23:03 UTC 1SEC_NOISEFAC3.MDF NF3 -> probably no data because of LV0 file size
27.1 23:03-11:05 UTC TRADEWIND_CU_UNLIMITED_UNCOMPRESSED.MDF NF6
28.1 11:05-14:17 UTC TRADEWIND_CU_UNLIMITED.MDF NF6, compressed
28.1-29.1 14:17-18:00 UTC TRADEWIND_CU_NF6_UNCOMPRESSED.MDF NF6
29.1-30.1 18:00-15:03 UTC 1SEC_NF6_UNCOMPRESSED.MDF NF6
30.1 15:10-23:42 UTC CU_SMALL_TINT_NF6_UNCOMPRESSED.MDF NF6, new chirp table
30.1-31.1 23:42-11:37 UTC CU_SMALL_TINT_NF3_UNCOMPRESSED.MDF NF3
31.1 11:37-22:28 UTC CU_SMALL_TINT_NF6_UNCOMPRESSED.MDF NF6
31.1 22:28-end UTC CU_SMALL_TINT2_NF6_UNCOMPRESSED.MDF NF6
"""

import sys
import datetime as dt
import time
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.NcWrite as nc
from larda.pyLARDA.spec2mom_limrad94 import spectra2moments, build_extended_container
import logging
import numpy as np

start_time = time.time()

log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]
begin_dates = [dt.datetime(2020, 1, 27, 3, 8, 6), dt.datetime(2020, 1, 27, 15, 7, 45),
               dt.datetime(2020, 1, 27, 23, 3, 4)]
end_dates = [dt.datetime(2020, 1, 27, 13, 3, 0), dt.datetime(2020, 1, 27, 21, 2, 0),
             dt.datetime(2020, 1, 27, 23, 59, 59)]
print('days with data', larda.days_with_data())
for begin_dt, end_dt in zip(begin_dates, end_dates):
    std_above_mean_noise = 6.0

    LIMRAD_Zspec = build_extended_container(larda, 'VSpec', begin_dt, end_dt,
                                            rm_precip_ghost=True,
                                            do_despeckle3d=False,
                                            estimate_noise=True,
                                            noise_factor=std_above_mean_noise
                                            )

    LIMRAD94_moments = spectra2moments(LIMRAD_Zspec, larda.connectors['LIMRAD94'].system_info['params'],
                                       despeckle=True,
                                       main_peak=True,
                                       filter_ghost_C1=True)
    for var in ['DiffAtt', 'ldr', 'bt', 'rr', 'LWP', 'MaxVel', 'C1Range', 'C2Range', 'C3Range', 'SurfRelHum',
                'Inc_El', 'Inc_ElA']:
        print('loading variable from LV1 :: ' + var)
        LIMRAD94_moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})

    LIMRAD94_moments['DiffAtt']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
                                                            LIMRAD94_moments['DiffAtt']['var'])
    LIMRAD94_moments['ldr']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
                                                        LIMRAD94_moments['ldr']['var'])

    path = f"/projekt2/remsens/data/campaigns/eurec4a/LIMRAD94/cloudnet_input_test"

    flag = nc.generate_cloudnet_input_LIMRAD94(LIMRAD94_moments, path, time_frame=f"{begin_dt:%H%M%S}-{end_dt:%H%M%S}")

    ########################################################################################################################

    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))