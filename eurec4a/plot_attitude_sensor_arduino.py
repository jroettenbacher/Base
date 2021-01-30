# plot time series of arduino attitude sensor
# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import re
import datetime
import glob
import numpy as np

# idea: use start times file to match filenames and set start time for datetime conversion
# define paths
input_path = "./data/ARDUINO"
output_path = "./plots"

# read in start files data
start_times = pd.read_csv(f"{input_path}/Arduino_start_times_read_in.TXT", sep="\t")
start_times["datetime"] = pd.to_datetime(start_times["datetime"], infer_datetime_format=True)
# list all files and match them with their start times
file_list = glob.glob(f"{input_path}/*_attitude_arduino.txt")
matched_times = pd.DataFrame(columns=["filename", "starttime"])
for f in file_list:
    date = re.search(r"\d{6}", f).group()
    date_dt = datetime.datetime.strptime(date, "%y%m%d")
    for i in range(len(start_times)):
        if start_times["datetime"].iloc[i].date() == date_dt.date():
            matched_times = matched_times.append({'filename': f, 'starttime': start_times["datetime"].iloc[i],
                                                  'date': date_dt.date()}, ignore_index=True)

# read in arduino data
date_dt = datetime.datetime(2020, 2, 2)
filename = f"{input_path}/{date_dt:%y%m%d}_attitude_arduino.TXT"
arduino = pd.read_csv(filename, sep=';', usecols=[0, 1, 2, 3])
arduino.columns = ["time", "heading_deg", "pitch_deg", "roll_deg"]  # change column names
# milliseconds from start should only increase in one file, if not something must have restarted the programm
# the header line is manually removed
# fix: if timestep n+1 is smaller than n add n to n+x
# find time step
timestep = arduino["time"].loc[arduino["time"].diff() < 0]
end = arduino["time"].index.max()
arduino["time"].loc[timestep.index.values[0]:end].update(arduino["time"].loc[timestep.index.values[0]:end])
# transform time from milliseconds from start to datetime
starttime = matched_times["starttime"].loc[matched_times["date"] == date_dt.date()]
arduino["time"] = pd.to_datetime(arduino["time"], unit='ms', origin=starttime.iloc[0])

# save daily files
import datetime as dt

def plot_attitude_sensor_arduino(filename, path, save_plot):
    # read in text file with pandas
    # adjust path to file
    # filename = "20191216_attitude_arduino.TXT"
    # read start time from file
    start_str = re.search(r"\d{6}_\d{6}", filename).group(0)
    # transform to datetime object
    start = datetime.datetime.strptime(start_str, "%y%m%d_%H%M%S")
    # filename = "20191216_141500_attitude_arduino.TXT"
    # read start time from file
    start_str = re.search(r"\d{8}_\d{6}", filename).group(0)
    # transform to datetime object
    start = dt.datetime.strptime(start_str, "%Y%m%d_%H%M%S")

    # read in file
    file = pd.read_csv(path+filename, sep=";",  # set seperator
                       usecols=[0, 1, 2, 3])  # select columns to use
    file.columns = ["time", "heading_deg", "pitch_deg", "roll_deg"]  # change column names
    # transform time from milliseconds from start to datetime
    file["time"] = pd.to_datetime(file["time"], unit='ms', origin=start)
    ts = file["time"].iloc[0]
    te = file["time"].iloc[-1]
    # cut off beginning and end if necessary
    # file = file[50:, :]  # cut of beginning
    # file = file[-50:, :]  # cut of ending

    # plot pitch, roll and  heading in separate plots because of scale difference
    plt.style.use("ggplot")
    plt.rcParams.update({'font.size': 16, 'figure.figsize': (16, 9)})
    hfmt = dates.DateFormatter('%d/%m/%y %H:%M')
    fig, ax = plt.subplots(nrows=3, sharex=True)
    ax[0].plot(file["time"], file["heading_deg"], 'r', label="Heading")
    ax[0].set_ylabel("Heading [deg]")
    ax[0].legend(loc='upper left')
    ax[0].set_title("Arduino Attitude Measurements")
    ax[1].plot(file["time"], file["pitch_deg"], 'b', label="Pitch")
    ax[1].set_ylabel("Pitch [deg]")
    ax[1].legend(loc='upper left')
    ax[2].plot(file["time"], file["roll_deg"], 'g', label="Roll")
    ax[2].set_ylabel("Roll [deg]")
    ax[2].legend(loc='upper left')
    # ax[2].xaxis.set_major_locator(dates.MinuteLocator())
    # ax[2].xaxis.set_major_formatter(hfmt)
    ax[2].set_xlabel("Date Time [UTC]")
    fig.autofmt_xdate()
    if save_plot == "y":
        os.chdir(r'C:\Users\Johannes\Studium\EUREC4A\data\plots_arduino')
        fig.savefig(filename.replace(".TXT", ".png"), dpi=300)
    ax[2].xaxis.set_major_locator(dates.MinuteLocator())
    ax[2].xaxis.set_major_formatter(hfmt)
    ax[2].set_xlabel("Date Time [UTC]")
    fig.autofmt_xdate()
    if save_plot == "y":
        fig.savefig(filename.replace(".TXT", ".png"))
        print("#########################################\n"
              "Figure saved to current working directory\n"
              "#########################################")
    else:
        fig.show()


if __name__ == "__main__":
    # filename = input("Insert file name: ")
    # path = input("Insert path to file: ")
    # save_plot = input("Save plot (y/n): ")
    filename = '200118_112848_attitude_arduino_plot.TXT'
    path = r"C:\Users\Johannes\Studium\EUREC4A\data\\"
    save_plot = 'y'
    filename = input("Insert file name: ")
    path = input("Insert path to file: ")
    save_plot = input("Save plot (y/n): ")
    plot_attitude_sensor_arduino(filename, path, save_plot)
