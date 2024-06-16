import os
import rasterio
from rasterio.enums import Resampling


root_path   = os.path.join(os.path.expanduser('~'), 'Data', 'aux')
in_file     = os.path.join(root_path, 'landWater2020.tif')
out_file    = os.path.join(root_path, 'landWater2020_1000m.tif')

# define target resolution, e.g. 1 km
in_res   = 10
out_res  = 3000


if __name__ == '__main__':

    with rasterio.open(in_file, dtype='float32') as dataset:

        # returns the (width, height) of pixels in the units of its coordinate reference system.
        xres, y_res = dataset.res

        # define rescaling factors
        scale_factor_x = in_res/out_res
        scale_factor_y = in_res/out_res

        # copy the meta data
        profile = dataset.profile.copy()

        # resample data to target shape,
        # see https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html#rasterio.io.BufferedDatasetWriter.read
        out_height, out_width = int(dataset.height * scale_factor_y), int(dataset.width * scale_factor_x)
        print(f'Resampling from ({dataset.height}, {dataset.width}) to ({out_height}, {out_width})')

        data = dataset.read(
            out_shape=(
                dataset.count,
                out_height, out_width
            ),
            out_dtype='float32',
            resampling=Resampling.bilinear
        )

        # scale image transform
        #"""
        transform = dataset.transform * dataset.transform.scale(
            (1 / scale_factor_x),
            (1 / scale_factor_y)
        )
        #"""

        #data[data>0.5]

        # update the meta data
        profile.update({"height": data.shape[-2],
                        "width": data.shape[-1],
                        "transform": transform,
                        "dtype": 'float32',
                        'compress': 'DEFLATE'})
        
    # write to file
    with rasterio.open(out_file, "w", **profile) as dataset:
        dataset.write(data)
    print(f'Done resampling. Writing to {out_file}.')