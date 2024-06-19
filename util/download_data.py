import os
import cdsapi
c = cdsapi.Client()

import numpy as np

# script to download ERA5 and GTSM data from Copernicus CDS, as well as further data from other sources

# to monitor your request queue on C3S, see:
# https://cds.climate.copernicus.eu/cdsapp#!/yourrequests?tab=form

store_at = os.path.expanduser('~/Data2')

# select the desired data in-line
dl_era5  = True	# ERA5
dl_gtsm  = True	# GTSM
dl_gesla = True	# GESLA3
dl_aux   = True	# auxiliary data

if __name__ == "__main__":

	if not os.path.exists(store_at): os.makedirs(store_at)
	
	if dl_era5:
		# fetch ERA5 data,
		# see https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

		era5_dir = os.path.join(store_at, 'ERA5', 'stormSurge_hourly_79_18')
		if not os.path.exists(era5_dir): os.makedirs(era5_dir)
		print(f'Downloading ERA5 data to {era5_dir}.')
		os.chdir(store_at)

		# for ERA5 downloads, the parameters need to be iterated (joint processing is too large)
		var_long  = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']
		var_short = ['u10', 'v10', 'msl']

		start, end, bins = 1979, 2018, 13
		yearIntervals = np.array_split(np.arange(start, end+1), bins)

		for idx, interval in enumerate(yearIntervals):
			if idx < 3: continue 
			for jdx, var_name in enumerate(var_long):
				print(f'Fetching {var_name} ERA5 data for years {interval}')
				
				c.retrieve(
					'reanalysis-era5-single-levels',
					{
						'product_type': 'reanalysis',
						'format': 'netcdf',
						'variable': [var_name],
						'month': [
							'01', '02', '03',
							'04', '05', '06',
							'07', '08', '09',
							'10', '11', '12',
						],
						'day': [
							'01', '02', '03',
							'04', '05', '06',
							'07', '08', '09',
							'10', '11', '12',
							'13', '14', '15',
							'16', '17', '18',
							'19', '20', '21',
							'22', '23', '24',
							'25', '26', '27',
							'28', '29', '30',
							'31',
						],
						'time': [
							'00:00', '01:00', '02:00',
							'03:00', '04:00', '05:00',
							'06:00', '07:00', '08:00',
							'09:00', '10:00', '11:00',
							'12:00', '13:00', '14:00',
							'15:00', '16:00', '17:00',
							'18:00', '19:00', '20:00',
							'21:00', '22:00', '23:00',
						],
						'year': [str(year) for year in interval],
					},
					os.path.join(era5_dir, "{}_{:02d}.nc".format(var_short[jdx], idx+1)))

	if dl_gtsm: 
		# fetch GTSM sea level data,
		# see https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-water-level-change-timeseries-cmip6
		# and https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-water-level-change-timeseries

		gtsm_dir = os.path.join(store_at, 'GTSM', 'reanalysis')
		if not os.path.exists(gtsm_dir): os.makedirs(gtsm_dir)
		os.path.join(gtsm_dir, 'reanalysis_surge_hourly_*_v1.nc')
		print(f'Downloading GTSM data to {gtsm_dir}.')
		os.chdir(gtsm_dir)

		for month in range(1,13): 
			print(f'Fetching GTSM data for month {month}')
			
			file_path = os.path.join(gtsm_dir, 'download{:02d}.zip'.format(month))
			c.retrieve(
				'sis-water-level-change-timeseries-cmip6',
				{
					'variable': 'storm_surge_residual',
					'experiment': 'reanalysis',
					'temporal_aggregation': [
						'10_min', 'hourly',
					],
					'year': [
						'1979', '1980', '1981',
						'1982', '1983', '1984',
						'1985', '1986', '1987',
						'1988', '1989', '1990',
						'1991', '1992', '1993',
						'1994', '1995', '1996',
						'1997', '1998', '1999',
						'2000', '2001', '2002',
						'2003', '2004', '2005',
						'2006', '2007', '2008',
						'2009', '2010', '2011',
						'2012', '2013', '2014',
						'2015', '2016', '2017',
						'2018',
					],
					'month': "{:02d}".format(month),
					'format': 'zip',
				},
				file_path)
			os.system(f'unzip {file_path} && rm {file_path}')
	
	if dl_gesla:

		gesla_dir = os.path.join(store_at, 'GESLA')
		if not os.path.exists(gesla_dir): os.makedirs(gesla_dir)
		os.chdir(gesla_dir)

		if os.path.isfile(os.path.join(gesla_dir, 'GESLA3.0_ALL.zip')):
			os.system('unzip GESLA3.0_ALL.zip && rm GESLA3.0_ALL.zip')
		else:
			# download GESLA data, lacks API access
			# see https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/gdj3.174
			raise NotImplementedError(f"Please download the data manually \
							 from https://www.icloud.com/iclouddrive/0tHXOLCgBBjgmpHecFsfBXLag#GESLA3 and place in {gesla_dir}")
		if not os.path.isfile(os.path.join(gesla_dir, 'GESLA3_ALL 2.csv')):
			raise NotImplementedError(f"Please download the data manually \
							 from https://www.icloud.com/iclouddrive/01a8u37HiumNKbg6CpQUEA7-A#GESLA3_ALL_2 and place in {gesla_dir}")
		
		os.system(f'mv GESLA3.0_ALL GESLA3')
		os.system(f'mv "GESLA3_ALL 2.csv" {gesla_dir}')
	
	if dl_aux: 
		# download auxiliary data from Zenodo

		aux_dir = os.path.join(store_at, 'aux')
		if not os.path.exists(aux_dir): os.makedirs(aux_dir)
		print(f'Downloading auxiliary data to {aux_dir}.')
		os.chdir(store_at)

		os.system('wget https://zenodo.org/api/records/11846592/files-archive')
		os.system('unzip files-archive && rm files-archive')
		os.system('unzip combined_gesla_surge.zip && rm combined_gesla_surge.zip')

		os.system(f'mv IBTrACS.ALL.v04r00.nc {aux_dir}')
		os.system(f'mv landWater2020_1000m.tif {aux_dir}')

		# for the original high-resolution land-sea masks, please see:
		# 	download land-sea mask, see https://user.iiasa.ac.at/~kinder/gfd/waterLand/readMe.html
		# 	os.system('wget https://user.iiasa.ac.at/~kinder/gfd/waterLand/landWater2020.tif')