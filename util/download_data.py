import os
import cdsapi
c = cdsapi.Client()

import numpy as np

# script to download ERA5 and GTSM data from Copernicus CDS, as well as further data from other sources

# to monitor your request queue on C3S, see:
# https://cds.climate.copernicus.eu/cdsapp#!/yourrequests?tab=form

# select the desired data in-line
dl_era5  = False	# ERA5
dl_gtsm  = False	# GTSM
dl_gesla = False 	# GESLA3
dl_lsm   = True		# land-sea mask

if __name__ == "__main__":

	if dl_era5:
		# fetch ERA5 data,
		# see https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

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
					"{}_{:02d}.nc".format(var_short[jdx], idx+1))

	if dl_gtsm: 
		# fetch GTSM sea level data,
		# see https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-water-level-change-timeseries-cmip6
		# and https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-water-level-change-timeseries

		for month in range(1,13): 
			print(f'Fetching GTSM data for month {month}')
			
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
				'download{:02d}.zip'.format(month))
	
	if dl_gesla:
		# download GESLA data, lacks API access
		# see https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/gdj3.174
		raise NotImplementedError("Please download the data manually \
							from https://www.icloud.com/iclouddrive/0tHXOLCgBBjgmpHecFsfBXLag#GESLA3")
	
	if dl_lsm: 
		# download land-sea mask, see https://user.iiasa.ac.at/~kinder/gfd/waterLand/readMe.html
		os.system('wget https://user.iiasa.ac.at/~kinder/gfd/waterLand/landWater2020.tif')