#!/usr/bin/env python
"""script to plot a height resolved frequency of occurence plot of dBZ values
input: Ze from larda
output: 2d frequency of occurence plot
author: Johannes Roettenbacher
"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import datetime as dt
import time
import numpy as np
import functions_jr as jr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import IndexFormatter


def calc_freq_of_occurrence_reflectivity(radar_ze, indices, bin_width=1):
    """Calculate the frequency of occurrence of reflectivity per height bin

    Args:
        radar_ze (larda container): linear reflectivity factor as read in with larda
        bin_width (int): which bin width should be used (optional, default=1)

    Returns: foc_array, hist_bins
        foc_array (np.array): array containing the frequencies in each reflectivity bin for each height bin, normalized
        to the total number of occurrences over the whole height;
        hist_bins (np.array): 1D array containing the bin edges including the rightmost bin

    """
    print("Computing histograms")
    start = time.time()
    # initialize a list to hold each histogram
    hist = []
    # calculate histogram for each height bin, remove nans beforehand
    # define histogram options
    hist_bins = np.arange(-60, 31, bin_width)  # bin edges including the rightmost bin (30)
    hist_range = (-60, 30)  # upper and lower bin boundary
    density = False  # compute the probability density function
    for i in range(radar_ze['var'].shape[1]):
        h_bin = h.lin2z(radar_ze['var'][indices, i])  # select height bin and convert to dBZ
        mask = radar_ze['mask'][indices, i]  # get mask for height bin
        h_bin[mask] = np.nan  # set -999 to nan
        tmp_hist, _ = np.histogram(h_bin[~np.isnan(h_bin)], bins=hist_bins, range=hist_range, density=density)
        hist.append(tmp_hist)

    # create an array to plot with pcolormesh by stacking the computed histograms vertically
    foc_array = np.array(hist, dtype=np.float)
    # set zeros to nan
    foc_array[foc_array == 0] = np.nan
    # normalize by count of all pixels
    foc_array = foc_array / np.nansum(foc_array)
    print(f"Computed histograms and created array for plotting in {time.time() - start:.2f} seconds")

    return foc_array, hist_bins


def plot_frequency_of_occurence_LIMRAD94(program, step, plot_path, larda_system):
    """Plot a 2D frequency of occurrence of reflectivity over height

    Args:
        program (list): list of chirp program numbers like 'P07'
        step (int): at which time step the plots should be made, allows for daily cycle plots (1, 2, 3, 4, 6, 8, 12, 24)
        plot_path (str): path where plot should be saved to
        larda_system (str): which larda system to use (LIMRAD94, LIMRAD94_cn_input), should have Ze and rg parameter

    Returns: 2D frequency of occurrence of reflectivity plot for specified chirp program(s)

    """

    start = time.time()
    # define durations of use for each chirp table (program)
    begin_dts = {'P09': dt.datetime(2020, 1, 17, 0, 0, 5), 'P06': dt.datetime(2020, 1, 30, 15, 30, 5),
                 'P07': dt.datetime(2020, 1, 31, 22, 30, 5)}
    end_dts = {'P09': dt.datetime(2020, 1, 27, 0, 0, 5), 'P06': dt.datetime(2020, 1, 30, 23, 42, 00),
               'P07': dt.datetime(2020, 2, 19, 23, 59, 55)}
    program_names = {'P09': "tradewindCU (P09)", 'P06': "Cu_small_Tint (P06)", 'P07': "Cu_small_Tint2 (P07)"}

    # get time chunk borders
    assert step in [1, 2, 3, 4, 6, 8, 12, 24], f"Step size is not a sensible value, {step}!" \
                                               f"Use an integer which 24 can be divided by!"
    steps = np.arange(0, 25, step)


    # get mean sensitivity limits
    mean_sl = jr.calc_sensitivity_curve(program)

    for p in program:
        assert p in program_names.keys(), f"Please use program codes like 'P07' to select chirptable! Not {p}! " \
                                          f"Check functions documentation to see which program corresponds to which " \
                                          f"chirptable"

        # define larda stuff
        system = larda_system
        begin_dt = begin_dts[p]
        end_dt = end_dts[p]
        plot_range = [0, 'max']
        # load LARDA
        larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
        # read in linear reflectivity factor from larda and convert to dBZ
        radar_ze = larda.read(system, "Ze", [begin_dt, end_dt], plot_range)
        print(f"Read in data from larda: {time.time() - start:.2f} seconds")

        # convert time stamps to datetime objects and create a pandas series
        dts = [h.ts_to_dt(t) for t in radar_ze['ts']]
        tmp_series = pd.Series(np.ones_like(radar_ze['ts']), index=dts)
        hours = tmp_series.index.hour  # get hours from index

        for i in range(len(steps)-1):
            # define a selector to only select hours in between step size
            selector = (steps[i] <= hours) & (hours < steps[i+1])
            # get indices of those time stamps
            indices = [tmp_series.index.get_loc(t) for t in tmp_series[selector].index]

            foc_array, hist_bins = calc_freq_of_occurrence_reflectivity(radar_ze, indices)

            # scale sensitivity limit to pcolormesh axis = 0-{number of hist_bins}, add lowest value in histogram bins
            mean_slh = h.lin2z(mean_sl['mean_slh'][p]) + np.abs(hist_bins[0])
            mean_slv = h.lin2z(mean_sl['mean_slv'][p]) + np.abs(hist_bins[0])

            figname = f"{plot_path}/RV-Meteor_{system}_freq_of_occurrence_{p}_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_{steps[i]}-{steps[i+1]}UTC.png"

            # create an array for the x and y tick labels
            height = radar_ze['rg']
            ylabels = np.floor(height) // 100 * 100 / 1000
            xlabels = hist_bins

            # create title
            title = f"Frequency of Occurrence of Reflectivity {system}" \
                    f"\nEUREC4A {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d} {steps[i]} - {steps[i+1]} UTC" \
                    f"\n Chirp program: {program_names[p]}"

            fig, ax = plt.subplots()
            im = ax.pcolormesh(foc_array, cmap='jet', norm=LogNorm())
            ax.plot(mean_slh, np.arange(len(height)), "k-", label="Horizontal Polarization")
            ax.plot(mean_slv, np.arange(len(height)), "r-", label="Vertical Polarization")
            fig.colorbar(im, ax=ax)
            fig.legend(title="Mean Sensitivity Limit", bbox_to_anchor=(0.5, -0.01), loc="lower center",
                       bbox_transform=fig.transFigure, ncol=2)
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.xaxis.set_major_formatter(IndexFormatter(xlabels))
            ax.yaxis.set_major_formatter(IndexFormatter(ylabels))
            ax.tick_params(which='minor', length=4, top=True, right=True)
            ax.tick_params(which='major', length=6)
            ax.xaxis.grid(True, which='major', color="k", linestyle='-', linewidth=1, alpha=0.5)
            ax.yaxis.grid(True, which='major', color="k", linestyle='-', linewidth=1, alpha=0.5)
            ax.set_xlabel("Reflectivity [dBZ]")
            ax.set_ylabel("Height [km]")
            ax.set_title(title)
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.2)
            fig.savefig(figname, dpi=300)
            plt.close()


if __name__ == '__main__':
    program = ['P06', 'P07', 'P09']
    steps = [3, 24]
    plot_path = "../plots/foc_LIMRAD94"
    larda_systems = ["LIMRAD94", "LIMRAD94_cn_input"]

    for larda_system in larda_systems:
        for step in steps:
            plot_frequency_of_occurence_LIMRAD94(program, step, plot_path, larda_system)
