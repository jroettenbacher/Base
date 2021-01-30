# LIMRAD94 Sensitivity Limit

During EUREC4A four different chirp tables were used. For the three most important ones the mean sensitivity limit for the whole duration of use is plotted. The data is filtered for rain and only non rainy profiles are used for the calculation of the mean sensitivity limit. The difference between filtered and unfiltered is also shown.

## Overview of Chirp Tables

| Chirp Table    | Program Number | From                 | To                   | Comment                   |
| -------------- | -------------- | -------------------- | -------------------- | ------------------------- |
| tradewindCU    | P09            | 17.01.2020           | 29.01.2020 18:00 UTC | Data gaps on the 27th     |
| Doppler1s      | P02            | 29.01.2020 18:00 UTC | 30.01.2020 15:03 UTC |                           |
| Cu_small_Tint  | P06            | 30.01.2020 15:10 UTC | 31.01.2020 22:28 UTC | Data gap 31.1 0-11:30 UTC |
| Cu_small_Tint2 | P07            | 31.01.2020 22:28 UTC | 29.02.2020           |                           |

### Screenshots of Chirp Tables

![tradewindCU](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\hardware\LIMRAD94\chirp_tables\tradewindCU.PNG)

![Doppler1s](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\hardware\LIMRAD94\chirp_tables\Doppler1s.PNG)

![Cu_small_Tint](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\hardware\LIMRAD94\chirp_tables\Cu_small_Tint.PNG)

![Cu_small_Tint2](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\hardware\LIMRAD94\chirp_tables\Cu_small_Tint2.PNG)

## Sensitivity Limits

#### Unfiltered Sensitivity Limit

![RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables_unfiltered](C:\Users\Johannes\PycharmProjects\Base\plots\sensitivity\RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables_unfiltered.png)

#### Rain filtered Sensitivity Limit

![RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables_filtered](C:\Users\Johannes\PycharmProjects\Base\plots\sensitivity\RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables_filtered.png)

#### Difference between filtered and unfiltered sensitivity limit

![RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables_filtered-unfiltered](C:\Users\Johannes\PycharmProjects\Base\plots\sensitivity\RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables_filtered-unfiltered.png)

Up to -8 dBZ difference can be observed between the filtered and the unfiltered sensitivity limits. Thus it can be said, that the sensitivity limit in non rainy conditions is lower, meaning more negative, than in rainy conditions. This is especially true for the lower range gates.

Chirp table Cu_small_Tint does not show any difference because it wasn't raining during its operation period. The different polarization channels also show different sensitivity changes due to rain. The vertical channel seems to be more affected than the horizontal channel.