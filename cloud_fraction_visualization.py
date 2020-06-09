
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_importer import WBandRadar



if __name__ == '__main__':
    
    # read cloud fraction from file
    cf_file = '/home/nrisse/PEA/Lectures/06_METRS/project/data/hydro_fraction_hourly.csv'
    cloud_fraction_df = pd.read_csv(cf_file)
    cloud_fraction_df['time'] = pd.to_datetime(cloud_fraction_df['time'])
    cloud_fraction_df.set_index('time', inplace=True)
    
    # extract dimensions from dataframe
    time_range = cloud_fraction_df.index
    rad = WBandRadar()
    rad.read_data(month='02', day='01', hour='00')
    range_bins = rad.range
    del rad
    
    # resample to 3 hourly means
    cloud_fraction_df_3h = cloud_fraction_df.resample('3H').mean()
    
    # calculate diurnal cycle (count, min, max, mean, std, 25/50/75 percentile)
    cloud_fraction_df_3h_diurnal = cloud_fraction_df_3h.copy()
    cloud_fraction_df_3h_diurnal['Time'] = cloud_fraction_df_3h_diurnal.index.map(lambda x: x.strftime("%H"))
    cloud_fraction_df_3h_diurnal = cloud_fraction_df_3h_diurnal.groupby(by='Time').describe().unstack()
    cloud_fraction_df_3h_diurnal.index.rename(['range_bin', 'stat', 'time'], inplace=True)
    
    # how to access data from data frame
    # data_frame['column_name=range_bin_from_0,1,...']['mean or std or ...']['time']
    
    
    # plot dirunal cylce of cloud fraction
    fig = plt.figure(figsize=(7, 10))
    ax = fig.add_subplot()
    ax.set_title('Cloud fraction from {} to {}'.format(time_range[0], time_range[-1]))
    
    # list of all ranges
    ix_list = np.full(shape=(3, len(range_bins)), fill_value=np.nan, dtype=object)
    ix_list[0, :] = [str(x) for x in range(0, len(range_bins))]  # range bins
    #ix_list[1, :] = 'mean'  # statistical measure
    ix_list[1, :] = 'min'  # statistical measure
    hours = [str(x).zfill(2) for x in range(0, 23, 3)]
    
    # colormap
    colors = cm.get_cmap('viridis', len(hours)).colors
    
    for i, hour in enumerate(hours):
        
        ix_list[2, :] = hour  # hour index
        ix = pd.MultiIndex.from_arrays(ix_list, names=['range_bin', 'stat', 'time'])
        ax.plot(cloud_fraction_df_3h_diurnal[ix], range_bins, color=colors[i], label='cloud fraction (' + hour + ' UTC)')
    
    ax.legend()
    ax.grid()
    ax.set_xlabel('Cloud fraction')
    ax.set_ylabel('Height [m]')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    
    plt.show()
    plt.savefig('/home/nrisse/PEA/Lectures/06_METRS/project/figures/hydro_fraction/hydro_fraction_min.png', dpi=100)
    
    
    # plot each hour in subplot and add standard deviations
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
    ax = axes.flatten()
    
    # list of all ranges
    ix_list = np.full(shape=(3, len(range_bins)), fill_value=np.nan, dtype=object)
    ix_list[0, :] = [str(x) for x in range(0, len(range_bins))]  # range bins
    hours = [str(x).zfill(2) for x in range(0, 23, 3)]
    
    for i, hour in enumerate(hours):
        
        ix_list[2, :] = hour  # hour index
        
        # get mean
        ix_list[1, :] = 'mean'
        ix = pd.MultiIndex.from_arrays(ix_list, names=['range_bin', 'stat', 'time'])
        cf_mean = cloud_fraction_df_3h_diurnal[ix].values
        
        # get standard deviation
        ix_list[1, :] = 'std'
        ix = pd.MultiIndex.from_arrays(ix_list, names=['range_bin', 'stat', 'time'])
        cf_std = cloud_fraction_df_3h_diurnal[ix].values

        ax[i].plot(cf_mean, range_bins, color='k')
        ax[i].plot(cf_mean + cf_std, range_bins, linestyle='--', color='k')
        ax[i].plot(cf_mean - cf_std, range_bins, linestyle='--',  color='k')
        
        ax[i].annotate(hour+' UTC', xy=(1, 1), xytext=(0.65, 0.9), xycoords='axes fraction')
        
    ax[0].set_ylabel('Height [m]')
    ax[4].set_ylabel('Height [m]')
    
    for i in range(4, 8):
        ax[i].set_xlabel('Cloud fraction')
    
    plt.savefig('/home/nrisse/PEA/Lectures/06_METRS/project/figures/hydro_fraction/hydro_fraction_sd_diurnal.png', dpi=100)
    