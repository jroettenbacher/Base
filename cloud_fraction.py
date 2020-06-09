
import pandas as pd
import numpy as np
from data_importer import WBandRadar



def cloud_fraction(ze, time, range_bins):
    """
    Calculate cloud fraction from radar reflectivity
    
    Definition of horizontal cloud fraction:
    cloud fraction = number of bins with signal at range / length of time interval
    
    Input
    ze           radar feflectivity as function of time and range
    time         corresponding time array
    range_bins   corresponding range array
    """
    
    n_time = len(time)
    n_levels = len(range_bins)  # exclude lower boundary

    # define array for cloud fraction
    cf = np.full(shape=(n_levels), fill_value=np.nan)
    
    for i in range(0, n_levels):
                
        # get number of bins at level that are not NaN
        n_signal = np.sum(~ze[:, i].mask)
        
        cf[i] = n_signal / n_time
    
    return cf


if __name__ == '__main__':
    
    # define time range for hourly cloud fraction calculation
    time_range = pd.date_range(start='2020-01-19 00:00', end='2020-02-11 23:59', freq='H')
    #time_range = pd.date_range(start='2020-01-19 00:00', end='2020-01-19 23:59', freq='H')
    
    # get number of height levels from radar data
    rad = WBandRadar()
    rad.read_data(month="02", day="01", hour="00")
    range_bins = rad.range
    cf_time_range = np.full(shape=(len(rad.range), len(time_range)), fill_value=np.nan)
    
    for i, time in enumerate(time_range):
        
        print('Calculate cloud fraction at time {}'.format(time))
        
        month = time.strftime('%m')
        day = time.strftime('%d')
        hour = time.strftime('%H')
    
        # read data
        try:
            rad = WBandRadar()
            rad.read_data(month=month, day=day, hour=hour)     
                        
        except IndexError:
            print('File not found at time {}'.format(time))
            rad = None
            
        if rad is not None:
        
            cf_time_range[:, i] = cloud_fraction(ze=rad.ze, time=rad.date_time, range_bins=rad.range)
        
        del rad
        
    # Save cloud fraction as data frame
    cloud_fraction_df = pd.DataFrame(index=time_range, columns=[str(x) for x in range(0, len(range_bins))])
    for i in range(0, len(range_bins)):
        cloud_fraction_df.loc[:, str(i)] = cf_time_range[i, :]
        
    # write hourly cloud fraction to file
    cloud_fraction_df.to_csv('/home/nrisse/PEA/Lectures/06_METRS/project/data/hydro_fraction_hourly.csv', index_label='time')
