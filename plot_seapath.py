#!\usr\bin\env python

"""Plot data from the Seapath motion sensor of RV-Meteor

stats available
- data availability

input:  1Hz Seapath data 17.1 - 27.1
        10Hz Seapath data 27.1 - 29.2
"""

import datetime as dt
import re
import pandas as pd
import numpy as np
import os
import glob

inpath = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP"
plotpath = "/projekt1/remsens/work/jroettenbacher/plots/seapath"

begin_dt = dt.date(2020, 1, 17)
end_dt = dt.date(2020, 2, 29)

# data read in
all_files = sorted(glob.glob(os.path.join(inpath + "/.*_DSHIP_seapath_*Hz.dat")))
file_list = []
for f in all_files:
    # match anything (.*) and the date group (?P<date>) consisting of 8 digits (\d{8})
    match = re.match(r"(?P<date>\d{8}).*", f)
    # convert string to datetime
    date_from_file = dt.datetime.strptime(match.group('date'), '%Y%m%d')
    if begin_dt <= date_from_file <= end_dt:
        file_list.append(f)
seapath = pd.concat(pd.read_csv(f, encoding='windows-1252', sep="\t", skiprows=(1, 2), index_col='date time')
                    for f in file_list)
seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
seapath.index.name = 'datetime'
seapath.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']