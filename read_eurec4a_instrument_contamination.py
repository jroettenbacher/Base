#!/usr/bin/env python

"""Script to read in information about instrument contamination from the DWD
input: txt file which was converted from rtf to txt by word
putput: csv table with date, start time, end time, note
output: txt file like filter_template.dat
"""

import re
import datetime as dt
import pandas as pd

def read_dwd_instrument_contamination(path, filename, encoding="utf-8"):
    f = open(f"{path}/{filename}", encoding=encoding)
    lines = f.readlines()

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

    for element in datetimes:
        m = re.match(pattern, element)
        startdates.append(dt.datetime.strptime(m.group('date1'), "%d.%m.%y"))
        enddates.append(dt.datetime.strptime(m.group('date2'), "%d.%m.%y"))
        starttimes.append(dt.time.fromisoformat(m.group('time1')))
        endtimes.append(dt.time.fromisoformat(m.group('time2')))

    table = pd.DataFrame({'startdate': startdates, 'enddate': enddates, 'starttime': starttimes, 'endtime': endtimes,
                          'note': notes})
    condition = []
    for i in range(len(table)):
        condition.append("Schornstein" in table["note"][i])

    table = table[condition].reset_index(drop=True)

    return table

if __name__ == 'main':
    path = "C:/Users/Johannes/Documents/Studium/Hiwi_Kalesse/HATPRO_flag_eurec4a"
    filename = "20200114_M161_Met-DWD_Instrument_Contamination.txt"
    table = read_dwd_instrument_contamination(path, filename)
