# plot time series of arduino attitude sensor
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import re
import datetime as dt


def plot_attitude_sensor_arduino(filename, path, save_plot):
    # read in text file with pandas
    # adjust path to file
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
    filename = input("Insert file name: ")
    path = input("Insert path to file: ")
    save_plot = input("Save plot (y/n): ")
    plot_attitude_sensor_arduino(filename, path, save_plot)
