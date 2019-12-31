import pandas as pd
import os
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import functions_jr as jr
# search for dates in all_files_mira and save them to a series
# split them up into year-month and day
# divide by count of days per month by usual days in each month

all_mira_files = open("all_files_mira.txt", mode='r')
lines = all_mira_files.readlines()
all_mira_files.close()
dates = []
for line in lines:
    dates.append(re.search(r'\d{8}', line).group(0))

dates_format = []
for date in dates:
    dates_format.append(datetime.datetime.strptime(date, '%Y%m%d'))
dates = pd.Series(dates_format)
counts_year = []
for year in range(2010, 2017):
    counts_year.append(np.count_nonzero([d for d in dates if d.year == year]))

plot_df = pd.DataFrame({'Year': np.arange(2010, 2017), 'Count': counts_year})
jr.set_presentation_plot_style()
plt.style.use('ggplot')
font = {'family': 'sans-serif', 'size': 24}
figure = {'figsize': [16, 9], 'dpi': 300}
plt.rc('font', **font)
plt.rc('figure', **figure)
plt.bar(plot_df['Year'], plot_df['Count'], color='blue')
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Number of MIRA35 Doppler Spectra Files per Year")
# plt.show()
plt.savefig('files_per_year.png')
plt.rcdefaults()