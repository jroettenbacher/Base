#!usr/bin/env python
"""
Plot of the jDMG Nachhaltigkeitsumfrage
author: Johannes Röttenbacher
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %% read in data and drop unusable columns
df = pd.read_csv("./data/jdmg/nachhaltigkeitsumfrage.csv", sep=",")
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
df_nutzen_long = df_nutzen.melt(var_name="Kategorie", value_name="Nutzen")
df_aufwand_long = df_aufwand.melt(var_name="Kategorie", value_name="Aufwand")
# take mean of each Kategorie
df_nutzen_long = df_nutzen_long.groupby("Kategorie", as_index=False).mean()
df_aufwand_long = df_aufwand_long.groupby("Kategorie", as_index=False).mean()

# %% merge data frames for plotting
df_plot = df_nutzen_long.merge(df_aufwand_long, on="Kategorie")

# %% plot

ax = df_plot.plot.scatter(x='Nutzen', y='Aufwand', alpha=0.5)
ax.grid()
for i, txt in enumerate(df_plot.index):
    ax.annotate(txt, (df_plot.Nutzen.iat[i], df_plot.Aufwand.iat[i]))
plt.show()
