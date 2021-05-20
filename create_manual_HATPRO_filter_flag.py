#!/usr/bin/env python

"""Script to read in information about instrument contamination from the DWD
input: txt file which was converted from rtf to txt by Word
putput: pd.Dataframe with startdate, enddate, start time, end time, note
output: txt file like filter_template.dat

Information:
While reading in the DWD file I noticed that the dates are sometimes not in consecutive order. I concluded that this was
an error and changed the dates to be in consecutive order. Thus I used ..._jr.txt as a source file. The original .rtf
file with non consecutive dates is untouched.
author: Johannes Röttenbacher
"""

import re
import datetime as dt
import pandas as pd
import numpy as np


def read_dwd_instrument_contamination(path, filename, encoding="utf-8"):
    """function to read in DWD text file about instrument errors during EUREC4A

    Args:
        path (str): path to file
        filename (str): filename
        encoding (str): optional encoding (="utf-8")

    Returns:
        df (pd.Dataframe): Data frame with start and end dates and times of errors which mention "Schornstein"

    """
    f = open(f"{path}/{filename}", encoding=encoding)
    lines = f.readlines()

    # list lines which mention the time range of the error and the error description
    datetimes = []
    notes = []
    for line_no in range(len(lines)):
        line = lines[line_no]
        if line.startswith("Zeitbereich:"):
            datetimes.append(line[13:])
        elif line.startswith("Bemerkungen:"):
            notes.append(lines[line_no+1])

    # create regex pattern to get dates and times from string
    pattern = re.compile(r"(?P<date1>\d{2}\.\d{2}\.\d{2}).*"
                         r"(?P<time1>\d{2}:\d{2}:\d{2}).*"
                         r"(?P<date2>\d{2}\.\d{2}\.\d{2}).*"
                         r"(?P<time2>\d{2}:\d{2}:\d{2}).*")

    # initialize lists for extracting information from datetimes strings
    startdates = []
    enddates = []
    starttimes = []
    endtimes = []

    # loop through datetime strings and extract start and end dates and times
    for element in datetimes:
        m = re.match(pattern, element)
        startdates.append(dt.datetime.strptime(m.group('date1'), "%d.%m.%y"))
        enddates.append(dt.datetime.strptime(m.group('date2'), "%d.%m.%y"))
        starttimes.append(dt.time.fromisoformat(m.group('time1')))
        endtimes.append(dt.time.fromisoformat(m.group('time2')))

    # create output data frame
    df = pd.DataFrame({'startdate': startdates, 'enddate': enddates, 'starttime': starttimes, 'endtime': endtimes,
                       'note': notes})
    # create list of boolean values to filter data frame for events which mention "Schornstein"
    condition = []
    for i in range(len(df)):
        condition.append("Schornstein" in df["note"][i])

    # return only events which mention "Schornstein"
    return df[condition].reset_index(drop=True)


def write_manual_filter(path, outfile, table):
    """write a .dat file with manual filter times for HATPRO.
     Still need to copy and paste it to the actual file!
     If an error stretches over two days and the next day has another entry, two lines with the same date are written.
     This might be a problem -> manually correct it in the file.

    Args:
        path (str): path where file should be written to
        outfile (str): name of filter file
        table (pd.DataFrame): table from read_dwd_instrument_contamination

    """
    f = open(f"{path}/{outfile}", "w")
    lines = []
    # set channel flags
    channels = "1 1 0"
    i = 0  # initialize counter to select rows from table
    while i < len(table):
        nn = 1
        start_date = table['startdate'][i]
        end_date = table['enddate'][i]
        # check the difference between consecutive start dates,
        # if there is no difference (same start date in consecutive rows) raise counter
        diffs = np.diff(table['startdate'][i:]).astype("double")
        try:
            while diffs[nn - 1] == 0:
                nn = nn + 1
        except IndexError:
            print("Reached end of table")

        j = 0  # initialize counter to write as many lines as needed per day
        while j < nn:
            if j == 0:
                # first line after starting a new row in the table
                if start_date == end_date:
                    date = dt.datetime.strftime(start_date, "%y%m%d")  # format date to yymmdd
                    start_time = table["starttime"][i]
                    # format start time to decimal hours
                    start_time_dec = f"{start_time.hour + start_time.minute / 60 + start_time.second / 3600:.2f}"
                    end_time = table["endtime"][i]
                    # format end time to decimal hours
                    end_time_dec = f"{end_time.hour + end_time.minute / 60 + end_time.second / 3600:.2f}"
                    # add line to list
                    lines.append(f"{date}  {nn} {start_time_dec.zfill(5)} {end_time_dec.zfill(5)} {channels}\n")

                elif start_date != end_date:
                    # write two lines if contamination stretches over two days
                    assert end_date == start_date + dt.timedelta(1), f"{end_date} is more than one day away from " \
                                                                     f"{start_date}! Cannot handle this case!"

                    date = dt.datetime.strftime(start_date, "%y%m%d")  # format date to yymmdd
                    start_time = table["starttime"][i]
                    # format start time to decimal hours
                    start_time_dec = f"{start_time.hour + start_time.minute / 60 + start_time.second / 3600:.2f}"
                    end_time_dec = "23.99"  # manually set end time in decimal hours
                    lines.append(f"{date}  {nn} {start_time_dec.zfill(5)} {end_time_dec.zfill(5)} {channels}\n")

                    date = dt.datetime.strftime(end_date, "%y%m%d")  # format date to yymmdd
                    start_time_dec = "00.00"  # manually set start time in decimal hours
                    end_time = table["endtime"][i]
                    # format end time to decimal hours
                    end_time_dec = f"{end_time.hour + end_time.minute / 60 + end_time.second / 3600:.2f}"
                    lines.append(f"{date}  1 {start_time_dec.zfill(5)} {end_time_dec.zfill(5)} {channels}\n")

            elif j > 0:
                # more lines if there are more contaminations in one day
                start_date = table['startdate'][i + j]
                end_date = table['enddate'][i + j]
                if start_date == end_date:
                    start_time = table["starttime"][i + j]
                    start_time_dec = f"{start_time.hour + start_time.minute / 60 + start_time.second / 3600:.2f}"
                    end_time = table["endtime"][i + j]
                    end_time_dec = f"{end_time.hour + end_time.minute / 60 + end_time.second / 3600:.2f}"
                    lines.append(f"          {start_time_dec.zfill(5)} {end_time_dec.zfill(5)} {channels}\n")

                elif start_date != end_date:
                    # write two lines if contamination stretches over two days
                    assert end_date == start_date + dt.timedelta(1), "end_date is more than one day away from " \
                                                                     "start_date! Cannot handle this case!"

                    start_time = table["starttime"][i + j]
                    start_time_dec = f"{start_time.hour + start_time.minute / 60 + start_time.second / 3600:.2f}"
                    end_time_dec = "23.99"
                    lines.append(f"          {start_time_dec.zfill(5)} {end_time_dec.zfill(5)} {channels}\n")

                    date = dt.datetime.strftime(end_date, "%y%m%d")  # format date to yymmdd
                    start_time_dec = "00.00"
                    end_time = table["endtime"][i + j]
                    end_time_dec = f"{end_time.hour + end_time.minute / 60 + end_time.second / 3600:.2f}"
                    lines.append(f"{date}  1 {start_time_dec.zfill(5)} {end_time_dec.zfill(5)} {channels}\n")

            j = j + 1
        i = i + j  # sum up counters to select next row with new date

    f.writelines(lines)
    f.close()
    print(f"File written to {path}/{outfile}")


def read_wras_contamination(path, file, encoding="utf-8"):
    """read in WRAS contamination file and process for HATPRO fitler creation

    Args:
        path (str): path to file
        file (str): filename
        encoding (str): optional encoding of input file (="utf-8")

    Returns:
        df (pd.DataFrame): Data frame with start and end dates and times of errors which mention " k"
    """
    df = pd.read_csv(f"{path}/{file}", sep=";", parse_dates=[0], dayfirst=True,
                     header=0, names=["startdate", "starttime", "endtime", "note"],  # define column names
                     encoding=encoding)

    # only select rows where comments mentions " k" for kilo, meaning CN is probably over 1000
    condition = []
    for i in range(len(df)):
        condition.append(" k" in df['note'][i])

    # add column end date for further processing
    df.insert(loc=1, column="enddate", value=df["startdate"])
    df["starttime"] = [dt.time.fromisoformat(t) for t in df["starttime"]]
    df["endtime"] = [dt.time.fromisoformat(t) for t in df["endtime"]]

    return df[condition].reset_index(drop=True)


if __name__ == '__main__':
    path = "C:/Users/Johannes/Documents/Hiwi_Kalesse/HATPRO_flag_eurec4a"
    file_dwd = "20200114_M161_Met-DWD_Instrument_Contamination_jr.txt"
    file_wras = "M161_WRAS_Stag_CPC_Jan-Feb2020_jr.csv"
    table1 = read_dwd_instrument_contamination(path, file_dwd)
    table2 = read_wras_contamination(path, file_wras)
    table = table1.append(table2).sort_values(axis=0, by=["startdate", "starttime"]).reset_index(drop=True)
    table.to_csv(f"{path}/smoke_flag_WRAS_DWD_eurec4a.csv", index=False)
    # manually delete the new line in the csv file!
    outfile = "filter_eurec4a.dat"
    write_manual_filter(path, outfile, table)  # write with both instrument flags
    outfile = "filter_eurec4a_dwd.dat"
    write_manual_filter(path, outfile, table1)  # write with only dwd flag
    outfile = "filter_eurec4a_wras.dat"
    write_manual_filter(path, outfile, table2)  # write with only wras flag
