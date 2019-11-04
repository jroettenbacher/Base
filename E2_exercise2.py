#!usr/bin/python3
# plot the Planck function for different temperatures
# author: Johannes Roettenbacher
# date: 04.11.2019


# import libraries
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

# defining constants
h = 6.6262 / (10 ** 34)  # Planck-constant
k = 1.3806 / (10 ** 23)  # Boltzmann-constant
c = 2.997925 * (10 ** 8)  # speed of light

# array approach, define temperature, wavelengths and frequency
T = np.array([5800, 1500, 288])  # temperature in Kelvin
frequencies = np.logspace(start=6, stop=18, num=10000)  # frequency in Herz
wavelengths = np.arange(start=0.01, stop=100.01, step=0.01) / (10 ** 6)  # wavelengths in meter

# frequency
B_freq = np.empty((len(frequencies), len(T)))  # initiate array with size of T x frequencies
B_freq_wien = np.empty((len(frequencies), len(T)))  # initiate array with size of T x frequencies
B_freq_rayleigh = np.empty((len(frequencies), len(T)))  # initiate array with size of T x frequencies
# loop through all combinations of temperature and wavelength and calculate spectral radiance with Planck's Law
for j in range(len(T)):
    for i in range(len(frequencies)):
        B_freq[i, j] = 2 * h * (frequencies[i] ** 3) / (c ** 2) * (1 / (np.exp(h * frequencies[i] / (k * T[j])) - 1))
        # Wien's approximation
        B_freq_wien[i, j] = 2 * h * (frequencies[i] ** 3) / (c ** 2) * np.exp(-h * frequencies[i] / (k * T[j]))
        # Rayleigh-Jeans approximation
        B_freq_rayleigh[i, j] = 2 * k * (frequencies[i] ** 2) * T[j] / (c ** 2)

# plot data frequency
plt.style.use('ggplot')  # which plotting style to use
for i in range(len(T)):
    fig1, ax1 = plt.subplots()
    ax1.plot(frequencies/1e9, B_freq[:, i], label=f"{T[i]} K")
    ax1.plot(frequencies/1e9, B_freq_wien[:, i], label=f"Wien's {T[i]} K")  # plot Wien's approximation
    ax1.plot(frequencies/1e9, B_freq_rayleigh[:, i], label=f"Rayleigh-Jeans {T[i]} K")  # plot Rayleigh-Jeans approximation
    ax1.set_yscale('log')  # set log scale
    ax1.set_xscale('log')
    ax1.tick_params(top=True, left=True, right=True, direction='in')  # where to place ticks which show inward
    ax1.tick_params(which='minor', top=True, bottom=True, left=False, direction='in')  # where to place minor ticks which show inward
    ax1.set_ylim(1e-20, max(B_freq[:, 0])*1e1)  # set limit for y axis
    ax1.set_xlim(min(frequencies)/1e9, max(frequencies)/1e9)  # set limit for x axis
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # format x ticks labels
    ax1.legend(title='Temperature and Approximation')  # add legend to plot
    ax1.set_title("Planck's Law \nas a Function of Frequency")  # set plot title
    ax1.set_xlabel(r"Frequency ($GHz$)")  # set x axis label
    ax1.set_ylabel(r"Radiance $B_\nu \, (W \, m^{-2} \, s \, sr^{-1})$")  # set y axis label
    ax1.fill_betweenx(y=[0, 1], x1=3, x2=300, color='r', alpha=0.2)
    ax1.annotate("Micro Waves", (1e-1, 1e-14))
    # plt.show()
    plt.savefig(f"plancks_law_frequency_{T[i]}.png", dpi=720)

# # wavelength
# B = np.empty((len(wavelengths), len(T)))  # initiate array with size of T x wavelengths
# B_wien = np.empty((len(wavelengths), len(T)))
# B_rayleigh = np.empty((len(wavelengths), len(T)))
# # loop through all combinations of temperature and wavelength and calculate spectral radiance with Planck's Law
# for j in range(len(T)):
#     for i in range(len(wavelengths)):
#         B[i, j] = 2 * h * (c ** 2) / (wavelengths[i] ** 5) / (np.exp(h * c / (k * wavelengths[i] * T[j])) - 1)
#         # Wien's approximation
#         B_wien[i, j] = 2 * h * (c ** 2) * np.exp(-h * c / k / wavelengths[i] / T[j]) / (wavelengths[i] ** 5)
#         # Rayleigh-Jeans approximation
#         B_rayleigh[i, j] = 2 * c * k * T[j] / (wavelengths[i] ** 4)
#
# # plot data wavelength
# for i in range(len(T)):
#     fig, ax = plt.subplots()
#     ax.plot(wavelengths * 1e6, B[:, i] / 1e6, label=f"{T[i]} K")  # convert wavelength to my meter and B per my meter
#     ax.plot(wavelengths * 1e6, B_wien[:, i] / 1e6, label=f"Wien's {T[i]} K")  # plot Wien's approximation
#     ax.plot(wavelengths * 1e6, B_rayleigh[:, i] / 1e6,
#             label=f"Rayleigh-Jeans {T[i]} K")  # plot Rayleigh-Jeans approximation
#     ax.tick_params(top=True, left=True, right=True, direction='in')  # where to place ticks which show inward
#     ax.tick_params(which='minor', top=True, left=False, direction='in')  # where to place minor ticks which show inward
#     ax.set_yscale('log')  # set log scale
#     ax.set_xscale('log')
#     ax.set_ylim(1, max(B[:, 0]/1e6)*1e1)  # set limit for y axis
#     ax.set_xlim(min(wavelengths)*1e6, max(wavelengths)*1e6)  # set limit for x axis
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # format x ticks labels
#     ax.legend(title='Temperature and Approximation')  # add legend to plot
#     ax.set_title("Planck's Law as a Function of Wavelength")  # set plot title
#     ax.set_xlabel(r"Wavelength ($\mu$m)")  # set x axis label
#     ax.set_ylabel(r"Spectral Radiance $B_\lambda \, (W \, m^{-2} \, sr^{-1} \, \mu m^{-1})$")  # set y axis label
#     ax.fill_betweenx(y=[0, 1e9], x1=1e2, x2=1e3, color='r', alpha=0.2)
#     ax.annotate("Micro Waves", (10, 1e3))
#     plt.show()
#     plt.savefig(f"plancks_law_wavelength_{T[i]}.png", dpi=720)
