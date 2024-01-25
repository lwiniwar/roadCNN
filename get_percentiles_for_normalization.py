from osgeo import gdal
import numpy as np


ds = gdal.Open(r"D:\lwiniwar\data\emerge\rasterdata\harmonic_parameters\cut_lwin\features_ndsm.vrt")

ranges = []
for bandid in range (1,11):
    band = ds.GetRasterBand(bandid)
    arr = band.ReadAsArray()
    arr[arr == band.GetNoDataValue()] = np.nan
    print(f"Band {bandid}:")
    perc = np.nanpercentile(arr.flatten(), [2, 98])
    print(perc)
    ranges.append(perc)

ranges = np.array(ranges)
print(",".join([f'{d:.2f}'for d in ranges[:, 0]]))
print(",".join([f'{d:.2f}'for d in ranges[:, 1]]))
