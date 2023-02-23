import sys

import numpy as np
import torch.utils.data
from osgeo import gdal, ogr
from itertools import permutations
import tqdm
import scipy.ndimage as simg
gdal.UseExceptions()

from torch.utils.data import Dataset

eps = 1e-6


class RoadDataset(Dataset):
    def __init__(self, data_vrt, label_vrt=None, polygon_file=None,
                 augmentation=True, overlap=32,
                 k_split_approx=10, num_channels=3, min_road_pixels=10, dilate_iter=0,
                 minv=None, maxv=None):
        super(RoadDataset, self).__init__()
        self.dilate_iter = dilate_iter
        self.num_channels = num_channels
        self.augmentation = augmentation
        self.data_vrt = str(data_vrt)
        self.label_vrt = str(label_vrt) if label_vrt is not None else None

        dataset = gdal.Open(self.data_vrt, gdal.GA_ReadOnly)
        # print("Calculating dataset statistics...")
        # stats = dataset.GetRasterBand(1).GetStatistics(0, 1)
        # print(f"Will normalize values from {stats[0]} - {stats[1]} to 0 - 1")
        self.minv = minv
        self.maxv = maxv
        self.xsize, self.ysize, self.rsize = dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount

        self.size = 224
        self.overlap = overlap
        self.mov = self.size - self.overlap
        self.num_x = self.xsize // (self.mov) - 1
        self.num_y = self.ysize // (self.mov) - 1
        num_ds = self.num_x * self.num_y

        boxes = self.spatial_split(k_split_approx)
        self.num_k = len(boxes)
        # run through tiles and identify the ones with all no-data
        self.lookup_list = [list() for i in range(len(boxes))] #list(range(num_ds))
    
        self.geotransform = dataset.GetGeoTransform()
        print("Origin = ({}, {})".format(self.geotransform[0], self.geotransform[3]))
        print("Pixel Size = ({}, {})".format(self.geotransform[1], self.geotransform[5]))
        self.srs = dataset.GetProjection()

        channelset = gdal.Open(self.data_vrt, gdal.GA_ReadOnly)
        channelband = channelset.GetRasterBand(1)
        if self.label_vrt is not None:
            labelset = gdal.Open(self.label_vrt, gdal.GA_ReadOnly)
            labelband = labelset.GetRasterBand(1)
        elif min_road_pixels > 0:
            print(f"You requested to check for a minimum number of road pixels (n>{min_road_pixels}),"
                  f"but have not supplied a reference raster file. Will skip the check and run on all tiles.")

        if polygon_file:
            poly = ogr.Open(polygon_file)
            polylay = poly.GetLayer()
            allgeoms = ogr.Geometry(type=ogr.wkbGeometryCollection)
            for fi in range(polylay.GetFeatureCount()):
                polyfeat = polylay.GetFeature(fi)
                polygeom = polyfeat.GetGeometryRef()
                allgeoms.AddGeometry(polygeom)
            print(f"Found {allgeoms.GetGeometryCount()} polygons.")
            for i in tqdm.tqdm(range(num_ds), desc="Checking input data", colour='#cbac54'):
                xmul = i % self.num_x
                ymul = i // self.num_x
                posx = xmul * self.mov
                posy = ymul * self.mov
                pi = ogr.Geometry(ogr.wkbPoint)
                pi.AddPoint(self.geotransform[0] + self.geotransform[1] * (posx + self.size / 2),
                           self.geotransform[3] + self.geotransform[5] * (posy + self.size / 2))
                if pi.Within(allgeoms):
                    for boxid, box in enumerate(boxes):
                        if posx >= box[0] and posx < box[1] and posy >= box[2] and posy < box[3]:
                            # check if tile has no roads
                            if self.label_vrt is None or np.count_nonzero(
                                    labelband.ReadAsArray(xoff=posx, yoff=posy, win_xsize=self.size, win_ysize=self.size)
                            ) > min_road_pixels:
                                featpatch = channelband.ReadAsArray(xoff=posx, yoff=posy, win_xsize=self.size,
                                                                    win_ysize=self.size)
                                if np.count_nonzero(featpatch == channelband.GetNoDataValue()) < (self.size ** 2 // 2):
                                    self.lookup_list[boxid].append((posx, posy))
                                    break
        else:
            print("Assuming raster is bbox aligned and has no nodata values (no aoi polygon supplied)")
            sys.stdout.flush()
            for i in tqdm.tqdm(range(num_ds), desc="Checking input data", colour='#cbac54'):
                xmul = i % self.num_x
                ymul = i // self.num_x
                posx = xmul * self.mov
                posy = ymul * self.mov
                for boxid, box in enumerate(boxes):
                    if posx >= box[0] and posx < box[1] and posy >= box[2] and posy < box[3]:
                        # check if tile has no roads
                        if self.label_vrt is None or np.count_nonzero(
                                labelband.ReadAsArray(xoff=posx, yoff=posy, win_xsize=self.size, win_ysize=self.size)
                        ) > min_road_pixels:

                            featpatch = channelband.ReadAsArray(xoff=posx, yoff=posy, win_xsize=self.size, win_ysize=self.size)
                            if np.count_nonzero(featpatch == channelband.GetNoDataValue()) < (self.size**2 // 2):
                                self.lookup_list[boxid].append((posx, posy))
                                break
        dataset = None
        print(f"Found {sum([len(lookup) for lookup in self.lookup_list])} valid patches.")
        self.flat_list = [item for boxid, box in enumerate(self.lookup_list) for item in box]

    def __len__(self):
        return sum([len(box) for box in self.lookup_list]) * (1 if not self.augmentation else 8)
    
    def spatial_split(self, k=10):
        if k == 1:
            return [[
                0, self.xsize,
                0, self.ysize,
            ]]
        # overlay bbox with approx. k approx. square areas
        area = self.xsize * self.ysize
        square_side = int(np.sqrt(area / k))
        #print(f"Area would fit into {k} squares of {square_side} px side length.")
        # find closest multipliers
        x_num = int(self.xsize/square_side)
        x_left = self.xsize - (x_num * square_side)
        #print(f"In x-dimension, {x_num} of these squares fit, leaving {x_left} pixel(s) uncovered")
        x_side = square_side + x_left // x_num
        #print(f"The size is therefore adjusted to {x_num} squares of length {x_side} (now leaving {self.xsize - (x_num * x_side)} pixel(s) uncovered)")
        y_num = k // x_num
        #print(f"In y-dimension, we then need {y_num} segments to result in k={x_num*y_num} / {k} parts")

        y_left = self.ysize - (y_num * square_side)
        #print(f"{y_num} tiles of {square_side} px would leave {y_left} pixel(s) uncovered")
        y_side = square_side + y_left // y_num
        #print(f"The size is therefore adjusted to {y_num} squares of length {y_side} (now leaving {self.ysize - (y_num * y_side)} pixel(s) uncovered)")
        #print(f"Rectangles are {x_side} x {y_side} pixels, and the whole area is {x_num} x {y_num} blocks ({x_num*y_num} blocks total).")
        #print(f"Total area is {x_side*x_num} x {y_side*y_num} pixels (of {self.xsize} x {self.ysize}).")

        outlist = []
        for curr_x in range(x_num):
            for curr_y in range(y_num):
                outlist.append([
                    curr_x*x_side, (curr_x+1) * x_side,
                    curr_y*y_side, (curr_y+1) * y_side,
                                ])

        return outlist

    def __getitem__(self, i):
        aug_id = 0
        if self.augmentation:
            aug_id = i // (len(self)/8)
            i = i // 8

        posx, posy = self.flat_list[i]
        # worker = torch.utils.data.get_worker_info()

        dataset = gdal.Open(self.data_vrt, gdal.GA_ReadOnly)
        # load item
        data = np.empty((self.num_channels, self.size, self.size), dtype=float)
        for bandId in range(self.num_channels):
            band = dataset.GetRasterBand(bandId + 1)
            banddata = band.ReadAsArray(xoff=posx, yoff=posy,
                                        win_xsize=self.size, win_ysize=self.size).astype(np.double)
            nanval = band.GetNoDataValue()
            banddata[banddata == nanval] = np.nanmean(banddata[banddata != nanval])  # mean imputation
            data[bandId, :, :] = np.clip((banddata - self.minv[bandId]) / (self.maxv[bandId] - self.minv[bandId]),
                                         0, 1)
            # data[bandId, :, :] = (banddata - self.minv[bandId]) / (self.maxv[bandId] - self.minv[bandId])

        # Data is normalized in [0, 1]
        # data = (data - self.minv) / (self.maxv - self.minv)

        if self.label_vrt:
            labelset = gdal.Open(self.label_vrt, gdal.GA_ReadOnly)
            labelband = labelset.GetRasterBand(1)
            label = labelband.ReadAsArray(xoff=posx, yoff=posy, win_xsize=self.size, win_ysize=self.size)#.transpose(2, 0, 1)
            if self.dilate_iter > 0:
                label = simg.binary_dilation(label, iterations=self.dilate_iter)
            label = label.astype(np.long)
        else:
            label = None

        # Data augmentation (https://doi.org/10.1371/journal.pone.0235487.g003)
        if aug_id == 1:
            data = np.flip(data, axis=1)  # left-right-flip
            if label: label = np.flip(label, axis=0)
        elif aug_id == 2:
            data = np.rot90(np.flip(data, axis=1), axes=(1,2))  # left-right-flip, then 90 deg rot
            if label: label = np.rot90(np.flip(label, axis=0), axes=(0,1))
        elif aug_id == 3:
            data = np.flip(data, axis=2)  # up-down-flip
            if label: label = np.flip(label, axis=1)
        elif aug_id == 4:
            data = np.rot90(np.flip(data, axis=2), axes=(1,2))  # up-down-flip, then 90 deg rot
            if label: label = np.rot90(np.flip(label, axis=1), axes=(0,1))
        elif aug_id > 4:
            data = np.rot90(data, k=aug_id-4, axes=(1, 2))  # 90/180/270 deg rot
            if label: label = np.rot90(label, k=aug_id-4, axes=(0, 1))  # 90/180/270 deg rot

        # Return the torch.Tensor values
        return (torch.from_numpy(data.copy()),
                torch.from_numpy(label.copy()) if label else torch.empty(()),
                (posx, posy))

    def write_results(self, res_array, target, nodata=-9999, dtype=gdal.GDT_Byte):
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(target, res_array.shape[1], res_array.shape[0], 1, dtype)
        outRaster.SetGeoTransform(self.geotransform)
        outband = outRaster.GetRasterBand(1)
        res_array[np.isnan(res_array)] = nodata
        outband.SetNoDataValue(nodata)
        outband.WriteArray(res_array)
        outRaster.SetProjection(self.srs)
        outband.FlushCache()


class SplitRoadDataset(RoadDataset):
    def __init__(self, iterate_sets=None, *args, **kwargs):
        super(SplitRoadDataset, self).__init__(*args, **kwargs)
        if iterate_sets is None:
            self.iterate_sets = []
        else:
            self.set_iterate_set(iterate_sets)


    def set_iterate_set(self, iterate_sets):
        self.iterate_sets = iterate_sets
        self.flat_list = [item for boxid, box in enumerate(self.lookup_list) for item in box if boxid in self.iterate_sets]

    def __len__(self):
        return sum([len(box) for boxid, box in enumerate(self.lookup_list) if boxid in self.iterate_sets]) * (1 if not self.augmentation else 8)

    def __getitem__(self, i):
        return super(SplitRoadDataset, self).__getitem__(i)



if __name__ == '__main__':



    dataset = gdal.Open(r"C:\Users\Lukas\Documents\Data\road-cnn-small\rapideye_2017\all.vrt", gdal.GA_ReadOnly)
    # dataset = gdal.Open(r"C:\Users\Lukas\Documents\Data\road-cnn-small\planet_2020\all.vrt", gdal.GA_ReadOnly)
    #
    # print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
    #                             dataset.GetDriver().LongName))
    # print("Size is {} x {} x {}".format(dataset.RasterXSize,
    #                                     dataset.RasterYSize,
    #                                     dataset.RasterCount))
    # print("Projection is {}".format(dataset.GetProjection()))
    # geotransform = dataset.GetGeoTransform()
    # if geotransform:
    #     print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    #     print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
    #
    # for bandId in range(dataset.RasterCount):
    #     print(f"\nBand #{bandId+1}:")
    #     print(f"==================================================")
    #     band = dataset.GetRasterBand(bandId+1)
    #     print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))
    #
    #     min = band.GetMinimum()
    #     max = band.GetMaximum()
    #     if not min or not max:
    #         (min,max) = band.ComputeRasterMinMax(True)
    #     print("Min={:.3f}, Max={:.3f}".format(min,max))
    #
    #     if band.GetOverviewCount() > 0:
    #         print("Band has {} overviews".format(band.GetOverviewCount()))
    #
    #     if band.GetRasterColorTable():
    #         print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))
    #
    # # start cutting 300x300 px areas
    # size = 300
    # overlap = 150
    # mov = size - overlap
    # num_x = dataset.RasterXSize // (mov) - 1
    # num_y = dataset.RasterYSize // (mov) - 1
    # print(f"Will have {num_x * num_y} tiles of size {size}x{size}px with an overlap of {overlap}px ({overlap/size*100.}%)")
    # for xmul in range(num_x):
    #     for ymul in range(num_y):
    #         posx = xmul * mov
    #         posy = ymul * mov
    #         data = np.empty((size, size, dataset.RasterCount), dtype=float)
    #         for bandId in range(dataset.RasterCount):
    #             band = dataset.GetRasterBand(bandId+1)
    #             data[:,:, bandId] = band.ReadAsArray(xoff=posx, yoff=posy,
    #                                    win_xsize=size, win_ysize=size)
    #         print(f"Got {size} x {size} px at {posx}, {posy}")