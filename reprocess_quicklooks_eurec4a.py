########################################################################################################################
#
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
import datetime
import sys

PATH_TO_LARDA = '/projekt1/remsens/work/jroettenbacher/Base/larda'

sys.path.append(PATH_TO_LARDA)

import pyLARDA
QUICKLOOK_PATH = '(projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/quicklooks)'


import logging
import numpy as np

import time
import pyLARDA.helpers as h

if __name__ == '__main__':


    start_time = time.time()

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    # gather argument
    begin_dt = datetime.datetime(int(args[2]), int(args[1]), int(args[0]), 0, 1)
    end_dt = datetime.datetime(int(args[2]), int(args[1]), int(args[0]), 23, 59)

    print(f'date :: {begin_dt:%Y%m%d}')

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.INFO)

    # Load LARDA
    larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

    TIME_SPAN_ = [begin_dt, end_dt]
    PLOT_RANGE_ = [0, 12000]

    limrad94_mom = dict()

    # load all variables
    limrad94_mom.update({var: larda.read("LIMRAD94_cni_hc_ca", var, TIME_SPAN_, PLOT_RANGE_) for var in
                         ['Ze', 'ldr', 'rr', 'sw', 'LWP', 'SurfRelHum', 'SurfWS', 'SurfTemp']})
    limrad94_mom.update({'VEL': larda.read("LIMRAD94_cni_hc_ca", "Vel_roll", TIME_SPAN_, PLOT_RANGE_)})

    # mask ldr since it was not calculated by this software (loading it from LV1 data)
    limrad94_mom['ldr']['var'] = np.ma.masked_where(limrad94_mom['Ze']['mask'], limrad94_mom['ldr']['var'])
    limrad94_mom['VEL']['var'] = np.ma.masked_where(limrad94_mom['Ze']['mask'], limrad94_mom['VEL']['var'])

    ########################################################################################################FONT=CYBERMEDIUM
    #
    #   ___  _    ____ ___ ___ _ _  _ ____    ____ ____ ___  ____ ____    _  _ ____ _  _ ____ _  _ ___ ____
    #   |__] |    |  |  |   |  | |\ | | __    |__/ |__| |  \ |__| |__/    |\/| |  | |\/| |___ |\ |  |  [__
    #   |    |___ |__|  |   |  | | \| |__]    |  \ |  | |__/ |  | |  \    |  | |__| |  | |___ | \|  |  ___]
    #
    plot_remsen_ql = True
    plot_radar_moments = False
    _FIG_SIZE = [9, 12]
    _DPI = 450

    # create folder for subfolders if it doesn't exist already
    h.change_dir(f'{QUICKLOOK_PATH}/')

    if plot_remsen_ql:
        fig, ax = pyLARDA.Transformations.remsens_limrad_quicklooks(limrad94_mom, plot_range=PLOT_RANGE_, timespan='24h')
        fig_name = f'{begin_dt:%Y%m%d}_QL_LIMRAD94.png'
        fig.savefig(fig_name, dpi=_DPI)
        print(f"   Saved to  {fig_name}")

    if plot_radar_moments:
        fig_name = f'{begin_dt:%Y%m%d}_Ze_LIMRAD94.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(limrad94_mom['Ze'], var_converter='lin2z', range_interval=PLOT_RANGE_)
        fig.savefig(fig_name, dpi=_DPI)
        print(f"   Saved to  {fig_name}")

        fig_name = f'{begin_dt:%Y%m%d}_VEL_LIMRAD94.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(limrad94_mom['VEL'], range_interval=PLOT_RANGE_, rg_converter=True)
        fig.savefig(fig_name, dpi=_DPI)
        print(f"   Saved to  {fig_name}")

        fig_name = f'{begin_dt:%Y%m%d}_sw_LIMRAD94.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(limrad94_mom['sw'], range_interval=PLOT_RANGE_, rg_converter=True)
        fig.savefig(fig_name, dpi=_DPI)
        print(f"   Saved to  {fig_name}")

        fig_name = f'{begin_dt:%Y%m%d}_skew_LIMRAD94.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(limrad94_mom['skew'], range_interval=PLOT_RANGE_, rg_converter=True)
        fig.savefig(fig_name, dpi=_DPI)
        print(f"   Saved to  {fig_name}")

        fig_name = f'{begin_dt:%Y%m%d}_kurt_LIMRAD94.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(limrad94_mom['kurt'], range_interval=PLOT_RANGE_, rg_converter=True)
        fig.savefig(fig_name, dpi=_DPI)
        print(f"   Saved to  {fig_name}")
