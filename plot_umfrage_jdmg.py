#!usr/bin/env python
"""
Plot of the jDMG Nachhaltigkeitsumfrage
author: Johannes Röttenbacher
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns

# %% read in data and drop unusable columns
base_dir = "C:/Users/Johannes/PycharmProjects/Base/data/jdmg"
df = pd.read_csv(f"{base_dir}/nachhaltigkeitsumfrage.csv", sep="\t", skipinitialspace=True)
# drop first and last two columns -> Zeitstempel and useless columns
# drop empty columns (Öffis zum Veranstaltungsort)
df = df.iloc[:, 1:-2].dropna(how='all', axis=1)
# select only specific rows
# df = df.iloc[[30, 31], :]

# %% split data frame in Aufwand and Nutzen
condition1 = ["Nutzen" in colname for colname in df.columns]
condition2 = ["Aufwand" in colname for colname in df.columns]
df_nutzen = df.loc[:, condition1]
df_aufwand = df.loc[:, condition2]
# change column names for later merge
df_nutzen.columns = [colname.replace("Nutzen: ", "") for colname in df_nutzen.columns]
df_aufwand.columns = [colname.replace("Aufwand: ", "") for colname in df_aufwand.columns]

# %% turn into long format
df_nutzen_long = df_nutzen.melt(var_name="Maßnahme", value_name="Nutzen")
df_aufwand_long = df_aufwand.melt(var_name="Maßnahme", value_name="Aufwand")
# remove leading whitespace in column Maßnahme
df_nutzen_long["Maßnahme"] = df_nutzen_long["Maßnahme"].str.strip()
df_aufwand_long["Maßnahme"] = df_aufwand_long["Maßnahme"].str.strip()
# take mean of each Maßnahme
df_nutzen_mean = df_nutzen_long.groupby("Maßnahme", as_index=False).mean().sort_values("Maßnahme")
df_aufwand_mean = df_aufwand_long.groupby("Maßnahme", as_index=False).mean().sort_values("Maßnahme")
# add column with standard deviation
df_nutzen_mean["std_nutzen"] = df_nutzen_long.groupby("Maßnahme").std().values
df_aufwand_mean["std_aufwand"] = df_aufwand_long.groupby("Maßnahme").std().values

# %% merge data frames for plotting
df_plot = df_nutzen_mean.merge(df_aufwand_mean, on="Maßnahme")
# add column with number of votes, Aufwand and Nutzen do not have the same amount of votes
df_plot["Anzahl"] = df_nutzen_long.groupby("Maßnahme", as_index=False).count().sort_values("Maßnahme").loc[:,"Nutzen"].values
# add a sum column for coloring and a number column that starts at 1
df_plot["sum"] = df_plot["Nutzen"] - df_plot["Aufwand"]
# sort by sum
df_plot = df_plot.sort_values(by="sum", ascending=False)
df_plot["Nummer"] = range(1, len(df_plot)+1)
df_plot.to_csv(f"{base_dir}/umfrageout.csv", index=False)

# %% plot
# define a Ampel like colormap
colors = cm.get_cmap("Spectral")
colors = ListedColormap(colors(np.linspace(0.3, 0.75, 10)))
# set aestetics
font = {'size': 14, 'sans-serif': ['Times New Roman']}
figure = {'figsize': [6, 8], 'dpi': 200}
plt.rc('font', **font)
plt.rc('figure', **figure)
# %% plot data
scale = 1  # scale the data to make errors bars better visible
ax = sns.scatterplot(data=df_plot*scale, x='Nutzen', y='Aufwand', hue="sum", palette=colors,
                     vmin=0, vmax=10, legend=False, s=75)
# add error bars in both directions
# ax.errorbar(df_plot.Nutzen*scale, df_plot.Aufwand*scale, xerr=df_plot.std_nutzen, yerr=df_plot.std_aufwand, fmt='none')
ax.grid()
ax.set_xlim([5*scale, 10*scale])
ax.set_ylim([0, 10*scale])
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.set_title("Einordnung der Maßnahmen zur \nVerbesserung der Nachhaltigkeit \ninnerhalb der DMG")
# annotate each dot with it's corresponding number
for i, txt in enumerate(df_plot.Nummer):
    ax.annotate(txt, (df_plot.Nutzen.iat[i]*scale, df_plot.Aufwand.iat[i]*scale))
# plt.show()
plt.savefig(f"{base_dir}/umfrage_nachhaltigkeit.png")
plt.close()
