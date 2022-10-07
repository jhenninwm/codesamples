# from datetime import date, timedelta
# from joblib import Parallel, delayed
from functools import partial
import geopandas as gpd
import multiprocessing
import pandas as pd
import numpy as np
import argparse
import calendar
import datetime as dt
import requests
import shapely
import pyproj
import shutil
import ee
import os
import time

from shapely.geometry import Point
from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry import Point

# Initialize ee kernel
ee.Initialize()


# ee.Authenticate()
# helpers for the download

def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 60)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def ConvertToFeature(shp):
    """
    Function to convert shapely polygons/geopandas DF to GEE Feature Collection

    Input:
        - shp: geopandas dataframe
    Output:
        - GEE FeatureCollection
    """

    features = []
    x, y = shp.exterior.coords.xy
    # print(x,y)
    cords = np.dstack((x, y)).tolist()

    g = ee.Geometry.Polygon(cords)
    feature = ee.Feature(g)
    features.append(feature)

    ee_object = ee.FeatureCollection(features)

    return ee_object


def GetDays(year, month):
    if month != 'all':
        r = list(calendar.monthrange(int(year), int(month)))[1]
        sdate = "-".join([str(year), str(month), str(1)])
        edate = "-".join([str(year), str(month), str(r)])
        return [sdate, edate]
    else:
        sdate = str(year) + "-" + str(1) + "-" + str(1)
        edate = str(year) + "-" + str(12) + "-" + str(31)
        return [sdate, edate]


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 60})
                .select('distance')
                .mask()
                .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER * 2 / 20)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                   .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


###############

'''
testing bounding box
'''
# https://pythoncharmers.com/blog/travel-distance-python-with-geopandas-folium-alphashape-osmnx-buffer.html

# Importing data
# shp = pd.read_csv('./data/COVID-19_HPSC_County_Statistics_Historic_Data.csv')
shp = pd.read_csv('covid_final_w.csv')
shp = shp[shp['Country/Region'] == 'US']
shp
# shp = shp.drop('Unnamed: 0', axis=1)

# Converting data to geopandas dataframe
gdf = gpd.GeoDataFrame(
    shp, geometry=gpd.points_from_xy(shp.Longitude, shp.Latitude),
    crs='epsg:4326')

gdf['PolygonID'] = [i for i in range(0, len(gdf))]
gdf.to_crs('EPSG:4326')
# buffer7km = (7/1.11)*.01
# gdf.geometry = gdf.geometry.buffer(buffer7km, cap_style=3)

gdf = gdf.sample(100)
print(gdf)
############### get the image from gee and downloads as zip


# Or change to the month you want the imagery from , 'all' does a yearly composite
month1 = "all"
ic = "COPERNICUS/S2_SR"
file_dir = 'sample_imagery'
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50

print('about to go to for loop')
count = 0
for col, row in gdf.iterrows():
    timest = time.perf_counter()
    print(col, 'col')
    # print(str(row.TimeStamp).split("/")[0], 'timestamp')
    print(row.Latitude, row.Longitude, 'coordinates')
    #    print(row.PolygonID)
    end = str(row.TimeStamp)
    start = str(pd.to_datetime(str(row.TimeStamp)) - dt.timedelta(89)).split(' ')[0]
    dates = [start, end]
    # year1 = str(row.TimeStamp).split("-")[0]
    # dates = GetDays(year1, month1)
    # start = dates[0]
    # end = dates[1]
    buffer6km = (5 / 1.11) * .01
    row.geometry = row.geometry.buffer(buffer6km, cap_style=3)
    cur_shp = row.geometry

    cur_shp = ConvertToFeature(cur_shp)
    geometry = ee.Geometry.Rectangle(list(row.geometry.bounds))

    # print(geometry)

    # if not os.path.isdir(os.path.join(file_dir, str(row.PolygonID) + "_" + str(row.ProvinceState) + "_" + str(row.TimeStamp))):
    # os.mkdir(os.path.join(file_dir, str(row.PolygonID) + "_" + str(row.ProvinceState) + "_" + str(row.TimeStamp))) #row.TileID
    # print('making new')
    # GET THE DOWNLOAD URL FOR THIS ONE - IT'S YOUR ORIGINAL IMAGE, CLOUDS & EVERYTHING

    s2_sr_cld_col = get_s2_sr_cld_col(cur_shp, start, end)
    raw_image = (s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median())
    # raw_image = add_cld_shdw_mask(raw_image)
    # raw_image = ee.ImageCollection(ic).filterBounds(cur_shp).filterDate(dates[0], dates[1]).sort('CLOUD_COVER').first()
    # raw_image = raw_image.clip(cur_shp)
    print(dates, 'dates')

    raw_image_zip_name = os.path.join(file_dir, str(row.PolygonID) + "_" + str(row.ProvinceState) + "_" + str(
        row.TimeStamp) + "_RI.zip")  # row.TileID
    raw_image_link = raw_image.select(['B4', 'B3', 'B2']).getDownloadURL(
        {'name': str(row.PolygonID), 'crs': 'EPSG:4326', 'fileFormat': 'GeoTIFF', 'region': geometry, 'scale': 10})
    print(raw_image_link, 'raw image link')

    r = requests.get(raw_image_link, allow_redirects=True)
    open(raw_image_zip_name, 'wb').write(r.content)
    sec = time.perf_counter() - timest
    print('This download took: ' + str(sec) + ' seconds.')

    count += 1
