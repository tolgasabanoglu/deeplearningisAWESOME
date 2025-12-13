import rasterio
import ee
import geemap
import geopandas as gpd
import numpy as np
import random
from tqdm import tqdm
%matplotlib inline
import os
import pandas as pd
import numpy as np
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # z-transformation
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics # Accuracy assessment
from multiprocessing import Pool
from osgeo import gdal
import geopy_ml_fun as geopy # Visualize and save map

# Authenticate and initialize Earth Engine API
ee.Authenticate()
ee.Initialize()

import geemap
import ee

# Set the Earth Engine asset ID for EnMap data
enmap_asset_id = 'projects/ee-tolgasabanoglu/assets/20230502_SPECTRAL_IMAGE'

# Load EnMap image to get the study area
enmap_image = ee.Image(enmap_asset_id)
study_area = enmap_image.geometry()

# Set the date range for Sentinel-2 data
start_date = '2023-04-01'
end_date = '2023-10-31'

# Load Sentinel-2 data for the specified date range, study area, and cloud cover
sentinel_data = (ee.ImageCollection('COPERNICUS/S2')
                 .filterBounds(study_area)
                 .filterDate(ee.Date(start_date), ee.Date(end_date))
                 .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 70))

# Calculate temporal metrics (median, imean, p25, p50, p75) for specified bands
temporal_metrics = sentinel_data.select(['B4', 'B3', 'B2', 'B8', 'B11', 'B12']).reduce(ee.Reducer.percentile([25, 50, 75], ['p25', 'p50', 'p75']).combine(ee.Reducer.mean(), '', True))

# Clip temporal metrics to the study area
temporal_metrics_clipped = temporal_metrics.clip(study_area)

# Display the study area on the map
Map = geemap.Map()
Map.centerObject(study_area, zoom=10)
Map.addLayer(study_area, {'color': 'FF0000'}, 'Study Area')

# Display EnMap data
Map.addLayer(enmap_image, {'bands': ['b40', 'b30', 'b20'], 'min': 0, 'max': 3000}, 'EnMap Data')

# Display temporal metrics with Red, Green, Blue, NIR, SWIR1, SWIR2 bands for visualization
temporal_metrics_vis = temporal_metrics_clipped.select(['B4_p25', 'B3_p25', 'B2_p25', 'B8_p25', 'B11_p25', 'B12_p25',
                                                       'B4_p50', 'B3_p50', 'B2_p50', 'B8_p50', 'B11_p50', 'B12_p50',
                                                       'B4_p75', 'B3_p75', 'B2_p75', 'B8_p75', 'B11_p75', 'B12_p75'])

# Rename bands to their corresponding spectral band names
spectral_band_names = ['Red_p25', 'Green_p25', 'Blue_p25', 'NIR_p25', 'SWIR1_p25', 'SWIR2_p25',
                       'Red_p50', 'Green_p50', 'Blue_p50', 'NIR_p50', 'SWIR1_p50', 'SWIR2_p50',
                       'Red_p75', 'Green_p75', 'Blue_p75', 'NIR_p75', 'SWIR1_p75', 'SWIR2_p75']

temporal_metrics_vis = temporal_metrics_vis.rename(spectral_band_names)

Map.addLayer(temporal_metrics_vis, {'min': 0, 'max': 3000}, 'Sentinel-2 Temporal Metrics (Visualization, Clipped)')

# Save temporal metrics to GEE with Red, Green, Blue, NIR, SWIR1, SWIR2 bands
bands_temporal_metrics = ['Red_p25', 'Green_p25', 'Blue_p25', 'NIR_p25', 'SWIR1_p25', 'SWIR2_p25',
                           'Red_p50', 'Green_p50', 'Blue_p50', 'NIR_p50', 'SWIR1_p50', 'SWIR2_p50',
                           'Red_p75', 'Green_p75', 'Blue_p75', 'NIR_p75', 'SWIR1_p75', 'SWIR2_p75']

sentinel_temporal_metrics = 'projects/ee-tolgasabanoglu/assets/sentinel_temporal_metrics_dry'
task_temporal_metrics = ee.batch.Export.image.toAsset(image=temporal_metrics_vis.select(bands_temporal_metrics),
                                                      description='sentinel_temporal_metrics',
                                                      assetId=sentinel_temporal_metrics,
                                                      scale=30)
task_temporal_metrics.start()

Map.addLayerControl()
Map
