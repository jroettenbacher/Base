#!/usr/bin/python
"""plot sensitivity curve for each chirptable of LIMRAD94 during eurec4a for both channels (vertical and horizontal
polarization), filtered by rain and not
- sensitivity limit is calculated for horizontal and vertical polarization
- for more information see LIMRAD94 manual chapter 2.6
- filter sensitivity during rain
author: Johannes Roettenbacher
"""

import sys

# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pyLARDA
import pyLARDA.helpers as h
import functions_jr as jr
import numpy as np
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    # define plot path
    plot_path = "/projekt1/remsens/work/jroettenbacher/plots/sensitivity"
    # Load LARDA
    larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
    system = "LIMRAD94"

    plot_range = [0, 'max']
    # names and program numbers of chirp tables
    chirptables = ("tradewindCU (P09)", "Cu_small_Tint (P06)", "Cu_small_Tint2 (P07)")
    programs = ["P09", "P06", "P07"]
    name = f'{plot_path}/RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables.png'

    begin_dts, end_dts = jr.get_chirp_table_durations(programs)
    stats = jr.calc_sensitivity_curve(programs, 'eurec4a')
    # read out height bins from each chirp and concat them to use as y-axis
    heights = dict()
    for p in programs:
        c1 = larda.read(system, "C2Range", [begin_dts[p]])['var'].data
        c2 = larda.read(system, "C2Range", [begin_dts[p]])['var'].data
        c3 = larda.read(system, "C3Range", [begin_dts[p]])['var'].data
        heights[p] = np.concatenate((c1, c2, c3))

    ####################################################################################################################
    # PLOTTING
    ####################################################################################################################
    plt.style.use('default')
    plt.rcParams.update({'figure.figsize': (16, 9)})

    # plot with all chirps and both polarizations
    fig, axs = plt.subplots(ncols=3, constrained_layout=True, sharey='all', sharex='all')
    for ax, program, i in zip(axs, programs, range(len(chirptables))):
        ax.plot(h.lin2z(stats['mean_slv'][program]), heights[program], label='vertical')
        ax.plot(h.lin2z(stats['mean_slh'][program]), heights[program], label='horizontal')
        ax.set_title(chirptables[i])
        ax.set_ylabel("Height [m]")
        ax.set_xlabel("Sensitivity Limit [dBZ]")
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
        ax.legend(title="Polarization")
    fig.suptitle(f"Unfiltered Mean Sensitivity Limit for LIMRAD94 \n"
                 f"Eurec4a - whole duration of chirptable use", fontsize=16)
    fig_name = name.replace(".png", "_unfiltered.png")
    fig.savefig(f'{fig_name}', dpi=250)
    logger.info(f'figure saved :: {fig_name}')
    plt.close()

    fig, axs = plt.subplots(ncols=3, constrained_layout=True, sharey='all', sharex='all')
    for ax, program, i in zip(axs, programs, range(len(chirptables))):
        ax.plot(h.lin2z(stats['mean_slv_f'][program]), heights[program], label='vertical')
        ax.plot(h.lin2z(stats['mean_slh_f'][program]), heights[program], label='horizontal')
        ax.set_title(chirptables[i])
        ax.set_ylabel("Height [m]")
        ax.set_xlabel("Sensitivity Limit [dBZ]")
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
        ax.legend(title="Polarization")
    fig.suptitle(f"Rain Filtered Mean Sensitivity Limit for LIMRAD94 \n"
                 f"Eurec4a - whole duration of chirptable use", fontsize=16)
    fig_name = name.replace(".png", "_filtered.png")
    fig.savefig(f'{fig_name}', dpi=250)
    logger.info(f'figure saved :: {fig_name}')
    plt.close()

    # plot the difference between filtered and unfiltered data
    fig, axs = plt.subplots(ncols=3, constrained_layout=True, sharey='all', sharex='all')
    for ax, program, i in zip(axs, programs, range(len(chirptables))):
        ax.plot(h.lin2z(stats['mean_slv_f'][program]) - h.lin2z(stats['mean_slv'][program]), heights[program],
                label='vertical')
        ax.plot(h.lin2z(stats['mean_slh_f'][program]) - h.lin2z(stats['mean_slh'][program]), heights[program]
                , label='horizontal')
        ax.set_title(chirptables[i])
        ax.set_ylabel("Height [m]")
        ax.set_xlabel("Sensitivity Limit [dBZ]")
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
        ax.legend(title="Polarization")
    fig.suptitle(f"Difference Between Rain Filtered and Unfiltered Mean Sensitivity Limit for LIMRAD94 \n"
                 f"Eurec4a - whole duration of chirptable use", fontsize=16)
    fig_name = name.replace(".png", "_filtered-unfiltered.png")
    fig.savefig(f'{fig_name}', dpi=250)
    logger.info(f'figure saved :: {fig_name}')
    plt.close()
