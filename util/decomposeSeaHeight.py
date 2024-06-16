import os
import glob
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
from datetime import datetime

from matplotlib import pyplot as plt
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d

from GeslaDataset import gesla
from utide import solve, reconstruct

from itertools import repeat
from multiprocessing import Pool

# define global variables and helper functions
failed_files, sparse_files = [], [] # lists, to which parallel function calls append to (not necessarily in original order)
ns_per_h = 3.6e12                   # nanoseconds per hour
get_sr   = lambda station, jdx, idx: (station.date_time[jdx]-station.date_time[idx]).values.item()/ns_per_h

def drop_duplicate_coords(xd, dim='date_time'):
	_, index = np.unique(xd[dim].values, return_index=True)
	return xd.isel(date_time=index)

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.path.join(os.path.expanduser('~'), 'Data'), type=str, help="path to your copy of the dataset")
parser.add_argument("--plots", dest="plots", action="store_true", help="whether to plot data or not") 
parser.add_argument("--process_files", dest="process_files", action="store_false", help="whether to pre-process data or not") 
parser.add_argument("--decompose", dest="decompose", action="store_false", help="whether to decompose sea level into its components or not") 
parser.add_argument("--merge_files", dest="merge_files", action="store_false", help="whether to merge data or not") 
parser.add_argument("--window_size", default=5, type=int, help="Size of the sliding window (in units of [h]), in stage 3 of pre=processing")
config = parser.parse_args()

def process_file(file_in, args):
    # fetch input and output file names
    exported_filenames  = args['exported_filenames']
    file_name           = os.path.basename(file_in)

    # skip readily processed files
    if file_name in exported_filenames: 
        print(f'Skipping {file_name}')
    #if file_name not in ['...']: return
    xarrayfied = geslaD.files_to_xarray([file_name])
    print(f'Started processing {file_name} at {datetime.now()}')

    try:
        # only consider samples from '1979-01-01' on, the beginning of GTSM C3S reanalysis product availability 
        station       = xarrayfied.sel(date_time=slice('1979-01-01', '2019-01-01')).load()
        is_coastal    = station.gauge_type == 'Coastal'

        # only consider gauges at coasts
        if not is_coastal: 
            print('Skipping. Non-coastal.')
            return
        # skip gauges with no records from 1979 on
        if len(station.date_time)<=1: 
            print('Skipping. No records within time period of interest.')
            return
        
        # run several checks on the selected tidal gauge time series:
        # check whether the dataset contains samples for every hour, TODO: this check is not perfect
        # check the time series' sampling frequency at the beginning, center and end --- to check probe for sufficiently dense sampling at any time period
        sample_ratio_too_infrequent = np.all([get_sr(station, jdx, idx) > 1 for idx, jdx in zip([0, len(station.date_time)//2 -1, -2],[1, len(station.date_time)//2, -1])])
        # check whether the dataset contains both on- and off-hour samples, i.e. sub-hour temporal resolution
        off_hourly      = (station.date_time.dt.minute != 0).sum() > 0 # e.g. port_aux_basques-665-can-meds measures at 11:30, 12:30, 13:30, ...
        on_n_off_hourly = (station.date_time.dt.minute == 0).sum() > 0 and (station.date_time.dt.minute != 0).sum() > 0 
        
        # check for duplicate time entries
        dropped_xr  = drop_duplicate_coords(station, dim='date_time')
        has_dupl    = len(station.date_time) != len(dropped_xr.date_time)
        if has_dupl: 
            # geslaD.files_to_xarray(...) may already be sufficiently take care of this
            print(f'Site {file_name} contains duplicate time stamps. Reducing.')
            station = dropped_xr
        
        if sample_ratio_too_infrequent: 
            print('Skipping. Samples not sufficiently frequent.')
            # TODO: some gauges may be fine except for a single jump in time, see e.g. aberdeen-abe-gbr-bod (1975/12/31 to 1980/01/01)
            # these currently get filtered, but this may be overly strict and may be handled differently in the future
            sparse_files.append(file_name)
            return # reject gauges sampled less often (e.g. only daily)
        if off_hourly or on_n_off_hourly: # for gauges at sub-hourly frequency, discard off-hour records
                # downsample any more frequent data to 1 h intervals, see
                # https://docs.xarray.dev/en/stable/generated/xarray.Dataset.resample.html
                # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
                
                print(f'Site {file_name} contains off- or sub-hourly records. Downsampling.')
                station = station.resample(date_time='H').mean()

        station_times = station.date_time                   # times are covering the entire timeline along all fetched stations
        station_level = station.sea_level.values.squeeze()  # ... so some stations may have NaN entries at times of no record
        station_lat   = station.latitude

        station_nan   = np.isnan(station_level)
        station_qc1   = np.logical_or(station.qc_flag.values.squeeze()==0, station.qc_flag.values.squeeze()==1) # use only 0 or 1, see np.unique(station.qc_flag)
        station_null  = station_level==station.null_value.values[0]                                             # typically -99.9999, but may differ

        station_valid = ~station_nan & station_qc1 & ~station_null
        station_valid = station_valid.squeeze()

        if config.decompose:
            # to isolate surge component from other components at tidal gauges:
            #   1. detrend, by subtracting annual mean sea-level variability (outcome is re-centered at 0)
            #   2. decompose via UTide into a) tide and b) non-tidal residuals,
            #      see https://github.com/wesleybowman/UTide
            #   3. denoise, by applying moving average window

            # 1. detrend mean sea level,
            # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html
            print('Detrending')
            valid_samples = station_valid.sum()
            if valid_samples <= 1: 
                print('Skipping. No valid samples left.')
                return
            if valid_samples < 24*31*356: 
                # don't detrend very short time series, but remove mean value (assumes hourly aggregates)
                detrend_sl = station_level[station_valid] - np.mean(station_level[station_valid])
            else:
                detrend_sl = detrend(station_level[station_valid], type='linear', bp=0, overwrite_data=False)

            # 2. decompose time series into tide & non-tidal components
            #   see: https://github.com/wesleybowman/UTide/blob/master/utide/_solve.py,
            #   OUT: coef, containing the amplitudes & phases from the fit, plus all configuration information
            #   which can be used in fct 'reconstruct' to generate a hindcast or forecast of the tides at the times specified in the time array
            print('Solving')
            coef = solve(
                station_times[station_valid],   # Time in days since `epoch`, time coordinates
                detrend_sl,                     # Sea-surface height, velocity component, etc. (also see arg time_series_v), put gauge values here
                lat=station_lat,                # Latitude in degrees (GESLA latitudes are given in [-90, 90])              
                nodal=False,                    # True (default) to include nodal/satellite corrections
                trend=True,                     # True (default) to include a linear trend in the model
                method="ols",                   # solvers: {'ols', 'robust'}
                conf_int="linear",              # confidence intervaL calculation technique, required for reconst
                Rayleigh_min=1.0,               # Minimum conventional Rayleigh criterion for automatic constituent selection; default is 1.
            )

            # which can be used in fct 'reconstruct' to generate a hindcast or forecast of the tides at the times specified in the time array
            
            # see: https://github.com/wesleybowman/UTide/blob/master/utide/_reconstruct.py
            #   OUT: Scalar time series is returned as `tide.h`; a vector series as `tide.u`, `tide.v`. Each is an ndarray with `np.nan` as the missing value.
            #   Most input kwargs are included: 'epoch', 'constit', 'min_SNR', and 'min_PE'.
            #   The input time array is included as 't_in', and 't_mpl'; the former is the original input time argument, 
            #   and the latter is the time as a datenum from '0000-12-31'.  If 'epoch' is 'python', these will be identical,
            #   and the names will point to the same array.

            print('Reconstructing')
            tide = reconstruct(
                    station_times[station_valid],   # Time in days since `epoch`, e.g.: obs.index
                    coef,                           # Data structure returned by `utide.solve`
                    epoch=None,
                    verbose=True,
                    constit=None,
                    min_SNR=2,
                    min_PE=0,
                )

            # remove tide component
            # scalar time series is returned as `tide.h`
            tide_height = tide['h']  
            non_tide = detrend_sl - tide_height

            # 3. filter the non-tidal residual
            # via a moving average filter (assuming hourly increments)
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html
            
            # impute invalid values with series means, then filter, only keep originally valid ones
            # (note: this is temporarily filling the gaps to not introduce spurrious correlations across long-term data gaps)
            print('Filtering')
            station_level[station_valid]  = non_tide
            station_level[~station_valid] = non_tide.mean()
            surge = uniform_filter1d(station_level, config.window_size, mode='reflect')[station_valid]
        else:
            # not applying msl-tide-surge decomposition
            print('Skipping detrending, decomposition and filtering.')
            surge = station_level[station_valid] 
        
        print('Exporting')
        station_id = geslaD.meta[geslaD.meta['filename'] == file_name].index.values

        if station_id.shape == station.longitude.shape:
            lon, lat = station.longitude.values, station.latitude.values
        else:
            lon = np.array([station.longitude.values.squeeze()[0]])
            lat = np.array([station.latitude.values.squeeze()[0]])

        # note: station_id can be utilized to access meta data via:
        #   self.geslaD = gesla.GeslaDataset(gesla_meta, gesla_path)
        #   self.geslaD.meta.iloc[902]
        xd = xr.Dataset(data_vars=dict(sea_level=(['station', 'date_time'], surge[None,:]),
                                    longitude=(['station'], lon),
                                    latitude =(['station'], lat)), 
                        coords=dict(date_time=station.date_time[station_valid].values,
                                    station=station_id), # station=xarrayfied.station),
                        attrs=dict(description="Filtered and re-processed GESLA-3 data.")
                        )			
        xd.to_netcdf(os.path.join(gesla_out, f'{file_name}.nc'))

        # plot data
        if config.plots:
            n_samples = 2500
            fig_dir = os.path.join(os.getcwd(), 'figures')
            fig, ax = plt.subplots()
            ax.plot(station_level[station_valid][:n_samples], label='raw')
            ax.plot(detrend_sl[:n_samples], label='detrend')
            plt.xlabel("time [h]")
            plt.ylabel("height [m]")
            plt.legend(loc='upper left')
            fig.tight_layout()
            plt.grid()
            fig.savefig(os.path.join(fig_dir, 'detrend_msl.pdf'), dpi=100)

            fig_dir = os.path.join(os.getcwd(), 'figures')
            fig, ax = plt.subplots()
            ax.plot(detrend_sl[:n_samples], label='height')
            ax.plot(tide_height[:n_samples], label='tide')
            plt.xlabel("time [h]")
            plt.ylabel("height [m]")
            plt.legend(loc='upper left')
            fig.tight_layout()
            plt.grid()
            fig.savefig(os.path.join(fig_dir, 'sea_height.pdf'), dpi=100)

            fig, ax = plt.subplots()
            ax.plot(non_tide[:n_samples], label='residual')
            ax.plot(surge[:n_samples], label='surge')
            plt.xlabel("time [h]")
            plt.ylabel("height [m]")
            plt.legend(loc='upper left')
            fig.tight_layout()
            plt.grid()
            fig.savefig(os.path.join(fig_dir, 'surge.pdf'), dpi=100)
    except:
        print(f'Failed processing file: {file_name}')
        failed_files.append(file_name)
    finally:
        # close read file
        xarrayfied.close()

def get_time(time_string):
    # split into date and time components, remove the decimal parts
    date, time  = time_string.split('.')[0].split('T')
    date, time  = [int(d) for d in date.split('-')], [int(t) for t in time.split(':')]
    timing      = datetime(*date, *time)
    return timing


if __name__ == '__main__':

    # set paths of input files
    gesla_path      = os.path.join(config.root, 'GESLA', 'GESLA3', '')
    gesla_meta      = os.path.join(config.root, 'GESLA', 'GESLA3_ALL 2.csv')
    gesla_out       = os.path.join(config.root, 'GESLA', f'netcdf_out_{config.window_size}h', '')
    outMergedFile   = f"combined_gesla_surge_{config.window_size}h.nc"
    if not os.path.exists(gesla_out): os.makedirs(gesla_out)

    # instantiate GESLA dataloader
    geslaD = gesla.GeslaDataset(gesla_meta, gesla_path)
    exported_filenames = [os.path.basename(f).split('.nc')[0] for f in glob.glob(os.path.join(gesla_out, '*.nc'))]

    if config.process_files:
        in_files = sorted(glob.glob(os.path.join(gesla_path, '*')))

        cpu_pool = 1
        print(f'Starting processing on a pool of {cpu_pool} workers.')
        if cpu_pool > 1:
            raise NotImplementedError('Not fully implemented yet, proceed with caution.')
            with Pool(processes=cpu_pool) as p:
                p.starmap(process_file, (in_files, repeat({'exported_filenames': exported_filenames})))
        else:
            for _, file_in in enumerate(tqdm(in_files)):
                process_file(file_in, {'exported_filenames': exported_filenames})
        
        plt.close('all')
        print(f'Done processing {len(in_files)-len(failed_files)} curated files.')
        print(f'Exported files to path {gesla_out}.')
        
        print(f'Failed processing files: {failed_files}')
        print('Some failed files end up with file size 0, prune those eg via: find . -type f -size 0 -print -delete')
        print(f'Skipped files due to infrequent time spacing: {sparse_files}')

    if config.merge_files:
        exported_files = sorted(glob.glob(os.path.join(gesla_out, '*.nc')))
        exported_non0_files = [path for path in exported_files if os.path.getsize(path) > 0]
        is_0 = list(sorted(set(exported_files) - set(exported_non0_files)))
        if len(is_0) > 0: print(f"Filtered {is_0} files of size 0.")

        print(f'Combining {len(exported_non0_files)} files into a single netcdf dataset of common time coordinates, started at {datetime.now()}.')
        # open multiple files, concat along time dimension & impute missing time points as NaN, see https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
        data  = xr.open_mfdataset(exported_non0_files, combine='nested', concat_dim="station", compat='equals', engine='netcdf4', use_cftime=True).load()
        print(f'Done opening multiple files, finished at {datetime.now()}.')

        save_file = os.path.join(config.root, outMergedFile)
        # eagerly read all data first, see https://github.com/pydata/xarray/issues/2912#issuecomment-485497398,
        # also consider https://docs.xarray.dev/en/latest/generated/xarray.save_mfdataset.html
        data.load().to_netcdf(save_file)
        print(f'Exported dataset to {save_file}, finished at {datetime.now()}.')

        print('Trying to read exported dataset.')
        xr_try = xr.open_dataset(filename_or_obj=save_file, engine='netcdf4')
        print('Successfully opened exported dataset.')
        print('Done')
