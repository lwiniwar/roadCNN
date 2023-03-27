from osgeo import gdal, ogr
import rtree
import cmcrameri as cmc
gdal.UseExceptions()

import numpy as np
from rich.progress import track, Progress

def get_density_within_region(roads, region, rtree_index):
    """
    The get_density_within_region function takes a list of roads and a region as input.
    It returns the density of road segments within the region, which is defined by the
    region's envelope (i.e., bounding box). The function first creates an index for
    the list of roads using Rtree, then iterates through each feature in the index to
    determine if it falls within the specified region. If so, it adds that feature's length to a running total.

    :param roads: Store the road data
    :param region: Specify the region of interest
    :param rtree_index: Find the features in roads that are within the bounding box of region
    :return: The density of roads within a region
    :doc-author: Trelent
    """
    xmin, xmax, ymin, ymax = region.GetEnvelope()
    length_sum = 0
    for featID in list(rtree_index.intersection((xmin, xmax, ymin, ymax))):
        item = roads[featID]
        roads_within = region.Intersection(item)
        if roads_within.GetGeometryType() > 0:  # not a null geometry
            if roads_within.GetGeometryCount() > 0:  # a multipart geometry
                for geom in roads_within:
                    length_sum += geom.Length()
            else:  # a singlepart geometry
                length_sum += roads_within.Length()
    return length_sum / region.GetArea()  # calculate density as m/m2

def get_density_moving_circle(roads, locations, radius, rtree_index):
    """
    The get_density_moving_circle function takes a list of roads, a list of locations, and the radius for the circle.
    It returns an array with the density at each location.

    :param roads: Get the road locations
    :param locations: Specify the locations where we want to calculate the density
    :param radius: Define the radius of the circle that is used to calculate the density
    :param rtree_index: Speed up the computation of the density
    :return: A matrix of densities for each location in the grid
    :doc-author: Trelent
    """
    zs = np.zeros((locations[0].shape[0], locations[1].shape[0]))
    with Progress() as prog:
        task_x = prog.add_task("[red] Loop over x", total=locations[0].shape[0])
        task_y = prog.add_task("[green] Loop over y", total=locations[1].shape[0])
        for (xi, x) in enumerate(locations[0]):
            prog.update(task_y, completed=0)
            for (yi, y) in enumerate(locations[1]):
                center = ogr.Geometry(ogr.wkbPoint)
                center.AddPoint(x, y)
                inters = center.Buffer(radius, 20)  # 20 points per quarter circle as approximation, 80 points for full circle
                dens = get_density_within_region(roads, inters, rtree_index)
                zs[xi, yi] = dens
                prog.update(task_y, advance=1)
            prog.update(task_x, advance=1)
    return zs

def write_raster(array, envelope, output_file):
    """
    The write_raster function writes a 2D array to a raster file.

    :param array: Pass the data to be written
    :param envelope: Specify the spatial extent of the raster
    :param output_file: Specify the path to the output file
    :return: The output_file
    :doc-author: Trelent
    """
    x_min, x_max, y_min, y_max = envelope
    rows, cols = array.shape
    x_res = (x_max - x_min) / rows
    y_res = (y_max - y_min) / cols

    driver = gdal.GetDriverByName("GTiff")
    out_raster = driver.Create(output_file, rows,cols, 1, gdal.GDT_Float32)
    out_raster.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(np.flipud(array.T))
    out_raster.FlushCache()

def main(args):
    """
    Function to create a road density map
    :param args: list. Element 0: input vector dataset. Element 1: pixel size for the output map. Element 2: search
    radius for the density calculation. Element 3: output path.
    :return: None
    """
    road_file = args[0]
    pixel_size = float(args[1])
    file = ogr.Open(road_file)
    lay = file.GetLayer()
    all_roads = []
    num_feats = lay.GetFeatureCount()
    rtree_index = rtree.index.Index(interleaved=False)
    tot_xmin = float("inf")
    tot_xmax = -float("inf")
    tot_ymin = float("inf")
    tot_ymax = -float("inf")
    for featid in range(num_feats):
        feat = lay.GetFeature(featid)
        geom = feat.GetGeometryRef()
        xmin, xmax, ymin, ymax = geom.GetEnvelope()
        tot_ymax = max(tot_ymax, ymax)
        tot_xmax = max(tot_xmax, xmax)
        tot_ymin = min(tot_ymin, ymin)
        tot_xmin = min(tot_xmin, xmin)
        rtree_index.insert(featid, (xmin, xmax, ymin, ymax))
        all_roads.append(geom.Clone())


    xlocs = np.arange(tot_xmin, tot_xmax, pixel_size)
    ylocs = np.arange(tot_ymin, tot_ymax, pixel_size)

    zz = get_density_moving_circle(all_roads, (xlocs, ylocs), float(args[2]), rtree_index)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(np.flipud(zz.T), cmap=cmc.cm.batlow)
    plt.colorbar()
    plt.axis('equal')
    plt.show()
    write_raster(zz, (tot_xmin, tot_xmax, tot_ymin, tot_ymax), args[3])


if __name__ == '__main__':
    main([
        r"C:\Users\Lukas\Documents\Data\road-cnn-small\cleaned roads\tile7\tile7 roads cleaned_NAD83UTM10N.shp",
        # r"C:\Users\Lukas\Documents\Data\road-cnn-small\20230206\tile7_res1_new3.shp",
        25.0,
        2000,
        'density_ref.tif'
        # 'density_test.tif'
    ])