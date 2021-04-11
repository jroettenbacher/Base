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
df = pd.read_csv(f"{base_dir}/nachhaltigkeitsumfrage.csv", sep=",")
# drop first and last two columns -> Zeitstempel and useless columns
# drop empty columns (Öffis zum Veranstaltungsort)
df = df.iloc[:, 1:-2].dropna(how='all', axis=1)

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
# take mean of each Maßnahme
df_nutzen_long = df_nutzen_long.groupby("Maßnahme", as_index=False).mean()
df_aufwand_long = df_aufwand_long.groupby("Maßnahme", as_index=False).mean()

# %% merge data frames for plotting
df_plot = df_nutzen_long.merge(df_aufwand_long, on="Maßnahme")
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
ax = sns.scatterplot(data=df_plot, x='Nutzen', y='Aufwand', hue="sum", palette=colors,
                     vmin=0, vmax=10, legend=False, s=75)
ax.grid()
ax.set_xlim([5, 10])
ax.set_ylim([0, 10])
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.set_title("Einordnung der Maßnahmen zur \nVerbesserung der Nachhaltigkeit \ninnerhalb der DMG")
# annotate each dot with it's corresponding number
for i, txt in enumerate(df_plot.Nummer):
    ax.annotate(txt, (df_plot.Nutzen.iat[i], df_plot.Aufwand.iat[i]))
# plt.show()
plt.savefig(f"{base_dir}/umfrage_nachhaltigkeit.png")
plt.close()
