"""
This routine calculates the radar moments for the RPG 94 GHz FMCW radar 'LIMRAD94' and generates a NetCDF4 file.
The generated files can be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored
    **NF   (float): number of standard deviations above mean noise floor

"""

import datetime
import logging
import numpy as np
import sys
import time

# LARDA_PATH = '/home/sdig/code/larda3/larda/'
LARDA_PATH = '/projekt1/remsens/work/jroettenbacher/Base/larda'

sys.path.append(LARDA_PATH)

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.SpectraProcessing as sp
import pyLARDA.Transformations as Trans
from larda.pyLARDA.NcWrite import rpg_radar2nc_eurec4a

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2020, Generates Calibrated RPG-Radar files for Cloudnetpy"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.3.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

########################################################################################################################
#
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
if __name__ == '__main__':

    start_time = time.time()

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    larda = pyLARDA.LARDA().connect('eurec4a')
    # larda = pyLARDA.LARDA().connect('leipzig_gpu')

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    # check argument/kwargs
    if 'date' in kwargs:
        date = str(kwargs['date'])
        begin_dt = datetime.datetime.strptime(date + ' 00:00:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 23:59:55', '%Y%m%d %H:%M:%S')
    else:
        date = '20200216'
        begin_dt = datetime.datetime.strptime(date + ' 00:00:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 00:59:55', '%Y%m%d %H:%M:%S')

    limrad94_settings = {
        'despeckle': True,  # 2D convolution (5x5 window), removes single non-zero values, very slow!
        'estimate_noise': True,  # estimating noise in spectra, when no fill_value is encountered
        'noise_factor': 6.0,  # noise_threshold = mean(noise) + noise_factor * std(noise)
        'despeckle2D': True,  # 2D convolution (5x5 window), removes single non-zero values,
        'ghost_echo_1': False,  # reduces the domain (Nyquist velocity) by ± 2.5 [m/s], when signal > 0 [dBZ] within 200m above antenna
        'ghost_echo_2': True,  # removes curtain like ghost echos
        'dealiasing': True,  # spectrum de-aliasing
        'heave_correction': True,  # correct for heave motion of ship
        'heave_corr_version': 'claudia',
        'add': False,  # add or subtract heave rate (move spectra to left or right)
        'shift': 0,  # number of time steps by which to shift seapath data of RV-Meteor
    }

    range_ = [0, 'max']
    limrad94_settings.update({'NF': float(kwargs['NF']) if 'NF' in kwargs else 6.0})
    site_ = kwargs['site'] if 'site' in kwargs else 'rv-meteor'
    limrad94_settings.update({'site': site_})
    PATH = kwargs['path'] if 'path' in kwargs else f'/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/tmp'

    log.info(f'Date: {begin_dt:%Y-%m-%d, %H:%M:%S} - {end_dt:%H:%M:%S}')
    TIME_SPAN_ = [begin_dt, end_dt]

    # new spectra processor v2
    radarZSpec = sp.load_spectra_rpgfmcw94(larda, TIME_SPAN_, **limrad94_settings)
    radarMoments = sp.spectra2moments(radarZSpec, larda.connectors['LIMRAD94'].system_info['params'], **limrad94_settings)
    radarMoments['VEL']['var'] = Trans.roll_mean_2D(radarMoments['VEL']['var'].copy(), 3, 'row')

    # load additional variables
    radarMoments.update({
        var: larda.read("LIMRAD94", var, TIME_SPAN_, range_) for var in [
            'ldr', 'DiffAtt', 'MaxVel', 'Azm', 'Elv', 'bt', 'rr', 'LWP', 'SurfRelHum'
        ]
    })

    # mask ldr since it was not calculated by this software (loading it from LV1 data)
    radarMoments['DiffAtt']['var'] = np.ma.masked_where(radarMoments['Ze']['mask'], radarMoments['DiffAtt']['var'])
    radarMoments['ldr']['var'] = np.ma.masked_where(radarMoments['Ze']['mask'], radarMoments['ldr']['var'])
    radarMoments['no_av'] = radarZSpec['no_av']  # copy from spectra dict
    radarMoments['rg_offsets'] = np.array(radarZSpec['rg_offsets'][:len(radarMoments['no_av'])]) + 1

    # convert from mm6 m-3 to dBZ
    radarMoments['Ze']['var'] = h.lin2z(radarMoments['Ze']['var'])
    radarMoments['Ze'].update({'var_unit': "dBZ", 'var_lims': [-60, 20]})

    # read in lat lon time series from RV Meteor
    dship = sp.read_dship(begin_dt.strftime("%Y%m%d"), cols=[0, 4, 5])
    dship_closest = sp.find_closest_timesteps(dship, radarMoments['Ze']['ts'])
    # extract lat lon arrays and save to dictionary to hand over to NcWrite.rpg_radar2nc_eurec4a()
    radarMoments['lat'] = dship_closest["SYS.STR.PosLat"].values
    radarMoments['lon'] = dship_closest["SYS.STR.PosLon"].values

    # write nc file
    flag = rpg_radar2nc_eurec4a(
        radarMoments,
        # f'{PATH}/limrad94/{begin_dt.year}/',
        f'{PATH}',
        larda_git_path=LARDA_PATH,
        version='python',
        **limrad94_settings
    )

    log.info(f'Exit with {flag},\ntotal elapsed time = {time.time() - start_time:.3f} sec.')
