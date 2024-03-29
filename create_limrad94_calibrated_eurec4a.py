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
import matplotlib.pyplot as plt
from pyLARDA.NcWrite import rpg_radar2nc_eurec4a
from distutils.util import strtobool
from functions_jr import find_bases_tops
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
        date = '20200211'
        begin_dt = datetime.datetime.strptime(date + ' 00:00:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 00:59:55', '%Y%m%d %H:%M:%S')

    PATH = kwargs['path'] if 'path' in kwargs else f'/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/' \
                                                   f'instruments/limrad94/cloudnet_input_testing'
    heave_corr_version = kwargs['heave_corr_version'] if 'heave_corr_version' in kwargs else 'ca'
    for_aeris = strtobool(kwargs['for_aeris']) if 'for_aeris' in kwargs else True

    limrad94_settings = {
        'despeckle': True,  # 2D convolution (5x5 window), removes single non-zero values, very slow!
        'estimate_noise': True,  # estimating noise in spectra, when no fill_value is encountered
        'noise_factor': 6.0,  # noise_threshold = mean(noise) + noise_factor * std(noise)
        'despeckle2D': True,  # 2D convolution (5x5 window), removes single non-zero values,
        'ghost_echo_1': False,  # reduces the domain (Nyquist velocity) by ± 2.5 [m/s], when signal > 0 [dBZ] within 200m above antenna
        'ghost_echo_2': True,  # removes curtain like ghost echos
        'dealiasing': True,  # spectrum de-aliasing
        'heave_correction': True,  # correct for heave motion of ship
        'heave_corr_version': heave_corr_version,
        'for_aeris': for_aeris,  # include more variables for upload to aeris
        'add': False,  # add or subtract heave rate (move spectra to left or right)
        'shift': 0,  # number of time steps by which to shift seapath data of RV-Meteor
    }

    range_ = [0, 'max']
    limrad94_settings.update({'NF': float(kwargs['NF']) if 'NF' in kwargs else 6.0})
    site_ = kwargs['site'] if 'site' in kwargs else 'rv-meteor'
    limrad94_settings.update({'site': site_})

    log.info(f'Date: {begin_dt:%Y-%m-%d, %H:%M:%S} - {end_dt:%H:%M:%S}')
    TIME_SPAN_ = [begin_dt, end_dt]

    # new spectra processor v2
    radarZSpec = sp.load_spectra_rpgfmcw94(larda, TIME_SPAN_, **limrad94_settings)
    radarMoments = sp.spectra2moments(radarZSpec, larda.connectors['LIMRAD94'].system_info['params'], **limrad94_settings)
    # add output from heave correction to radarMoments
    for var in ['heave_cor', 'heave_cor_bins', 'time_shift_array']:
        radarMoments[f'{var}'] = radarZSpec[f'{var}']
    if log.level < 20:
        # get hourly quicklooks
        plot_dt = begin_dt
        while plot_dt < end_dt:
            plot_interval = [plot_dt, plot_dt+datetime.timedelta(hours=1)]
            fig, ax = Trans.plot_timeheight2(radarMoments['VEL'], range_interval=[0, 3000], time_interval=plot_interval)
            plt.savefig(f"{PATH}/hourly_quicklooks/RV-Meteor_{plot_dt:%Y-%m-%d_%H}_mdv_cor_low_{heave_corr_version}.png")
            plt.close()
            fig, ax = Trans.plot_timeheight2(radarMoments['VEL'], range_interval=[3000, 6000], time_interval=plot_interval)
            plt.savefig(f"{PATH}/hourly_quicklooks/RV-Meteor_{plot_dt:%Y-%m-%d_%H}_mdv_cor_mid_{heave_corr_version}.png")
            plt.close()
            fig, ax = Trans.plot_timeheight2(radarMoments['VEL'], range_interval=[6000, 9000], time_interval=plot_interval)
            plt.savefig(f"{PATH}/hourly_quicklooks/RV-Meteor_{plot_dt:%Y-%m-%d_%H}_mdv_cor_high_{heave_corr_version}.png")
            plt.close()
            plot_dt = plot_dt + datetime.timedelta(hours=1)

    # read out mean Doppler velocity, replace -999 with nan, apply rolling mean, set nan to -999 and update mean Doppler
    # velocity in moments
    radarMoments['VEL_roll'] = radarMoments['VEL'].copy()  # create new container for averaged mean Doppler velocity
    vel = radarMoments['VEL']['var'].copy()
    vel[radarMoments['VEL']['mask']] = np.nan
    vel_mean = Trans.roll_mean_2D(vel, 3, 'time')
    vel_mean[radarMoments['VEL']['mask']] = -999
    radarMoments['VEL_roll']['var'] = vel_mean

    if log.level < 20:
        # get hourly quicklooks after rolling mean
        plot_dt = begin_dt
        while plot_dt < end_dt:
            plot_interval = [plot_dt, plot_dt + datetime.timedelta(hours=1)]
            fig, ax = Trans.plot_timeheight2(radarMoments['VEL_roll'], range_interval=[0, 3000],
                                             time_interval=plot_interval)
            plt.savefig(f"{PATH}/hourly_quicklooks/RV-Meteor_{plot_dt:%Y-%m-%d_%H}_mdv_cor_low_roll_{heave_corr_version}.png")
            plt.close()
            fig, ax = Trans.plot_timeheight2(radarMoments['VEL_roll'], range_interval=[3000, 6000],
                                             time_interval=plot_interval)
            plt.savefig(f"{PATH}/hourly_quicklooks/RV-Meteor_{plot_dt:%Y-%m-%d_%H}_mdv_cor_mid_roll_{heave_corr_version}.png")
            plt.close()
            fig, ax = Trans.plot_timeheight2(radarMoments['VEL_roll'], range_interval=[6000, 9000],
                                             time_interval=plot_interval)
            plt.savefig(f"{PATH}/hourly_quicklooks/RV-Meteor_{plot_dt:%Y-%m-%d_%H}_mdv_cor_high_roll_{heave_corr_version}.png")
            plt.close()
            plot_dt = plot_dt + datetime.timedelta(hours=1)

    # load additional variables
    radarMoments.update({
        var: larda.read("LIMRAD94", var, TIME_SPAN_, range_) for var in [
            'ldr', 'DiffAtt', 'MaxVel', 'Azm', 'Elv', 'bt', 'rr', 'LWP', 'SurfRelHum'
        ]
    })
    # add the uncorrected mean Doppler velocity again
    radarMoments['VEL_uncor'] = larda.read("LIMRAD94", 'VEL', TIME_SPAN_, range_)

    # mask ldr since it was not calculated by this software (loading it from LV1 data)
    radarMoments['DiffAtt']['var'] = np.ma.masked_where(radarMoments['Ze']['mask'], radarMoments['DiffAtt']['var'])
    radarMoments['ldr']['var'] = np.ma.masked_where(radarMoments['Ze']['mask'], radarMoments['ldr']['var'])
    radarMoments['no_av'] = radarZSpec['no_av']  # copy from spectra dict
    radarMoments['rg_offsets'] = np.array(radarZSpec['rg_offsets'][:len(radarMoments['no_av'])]) + 1

    # convert from mm6 m-3 to dBZ
    radarMoments['Ze']['var'] = h.lin2z(radarMoments['Ze']['var'])
    radarMoments['Ze'].update({'var_unit': "dBZ", 'var_lims': [-60, 20]})
    #mask LDR=-100 (-100 means there is a signal, but not clear enough to calculate LDR)
    radarMoments['ldr']['var'] = np.ma.masked_less_equal(radarMoments['ldr']['var'], -100)
    # add cloud bases and tops, and cloud mask
    _, radarMoments['cloud_bases_tops'] = find_bases_tops(radarMoments["Ze"]["mask"], radarMoments["Ze"]["rg"])
    radarMoments['hydrometeor_mask'] = ~radarMoments["Ze"]["mask"]

    if for_aeris:
        # read in lat lon time series from RV Meteor
        dship = sp.read_dship(begin_dt.strftime("%Y%m%d"), cols=[0, 4, 5])
        dship_closest = sp.find_closest_timesteps(dship, radarMoments['Ze']['ts'])
        # extract lat lon arrays and save to dictionary to hand over to NcWrite.rpg_radar2nc_eurec4a()
        radarMoments['lat'] = dship_closest["SYS.STR.PosLat"].values
        radarMoments['lon'] = dship_closest["SYS.STR.PosLon"].values

    # write nc file
    flag = rpg_radar2nc_eurec4a(
        radarMoments,
        f'{PATH}',
        larda_git_path=LARDA_PATH,
        version='python',
        **limrad94_settings
    )

    log.info(f'Exit with {flag},\ntotal elapsed time = {time.time() - start_time:.3f} sec.')
