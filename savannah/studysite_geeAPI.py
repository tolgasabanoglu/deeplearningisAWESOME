# --- Standard library imports ---
import os

# --- Third-party imports ---
import ee
import geemap
import numpy as np
import pandas as pd
from tqdm import tqdm

# (Optional / downstream ML & GIS imports kept for later use)
import rasterio
import geopandas as gpd
from osgeo import gdal
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from multiprocessing import Pool

import geopy_ml_fun as geopy  # Custom visualization utilities

# ===============================
# 1. Initialize Google Earth Engine
# ===============================

ee.Authenticate()
ee.Initialize()

# ===============================
# 2. User-defined parameters
# ===============================

# EnMap asset (defines study area)
ENMAP_ASSET_ID = 'projects/ee-tolgasabanoglu/assets/20230502_SPECTRAL_IMAGE'

# Sentinel-2 temporal range
START_DATE = '2023-04-01'
END_DATE = '2023-10-31'

# Cloud filtering threshold (%)
MAX_CLOUD_PERCENT = 70

# Export settings
EXPORT_SCALE = 30
EXPORT_ASSET_ID = 'projects/ee-tolgasabanoglu/assets/sentinel_temporal_metrics_dry'
EXPORT_DESCRIPTION = 'sentinel_temporal_metrics'

# Sentinel-2 bands of interest
S2_BANDS = ['B4', 'B3', 'B2', 'B8', 'B11', 'B12']  # Red, Green, Blue, NIR, SWIR1, SWIR2

# ===============================
# 3. Load EnMap image & study area
# ===============================

enmap_image = ee.Image(ENMAP_ASSET_ID)
study_area = enmap_image.geometry()

# ===============================
# 4. Load & filter Sentinel-2 data
# ===============================

sentinel_collection = (
    ee.ImageCollection('COPERNICUS/S2')
    .filterBounds(study_area)
    .filterDate(ee.Date(START_DATE), ee.Date(END_DATE))
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', MAX_CLOUD_PERCENT)
)

# ===============================
# 5. Temporal metrics computation
# ===============================

# Percentiles + mean
reducer = (
    ee.Reducer.percentile([25, 50, 75], ['p25', 'p50', 'p75'])
    .combine(ee.Reducer.mean(), '', True)
)

# Apply reducer
temporal_metrics = sentinel_collection.select(S2_BANDS).reduce(reducer)

# Clip to study area
temporal_metrics = temporal_metrics.clip(study_area)

# ===============================
# 6. Rename bands for clarity
# ===============================

renamed_bands = [
    'Red_p25', 'Green_p25', 'Blue_p25', 'NIR_p25', 'SWIR1_p25', 'SWIR2_p25',
    'Red_p50', 'Green_p50', 'Blue_p50', 'NIR_p50', 'SWIR1_p50', 'SWIR2_p50',
    'Red_p75', 'Green_p75', 'Blue_p75', 'NIR_p75', 'SWIR1_p75', 'SWIR2_p75'
]

band_order = [
    'B4_p25', 'B3_p25', 'B2_p25', 'B8_p25', 'B11_p25', 'B12_p25',
    'B4_p50', 'B3_p50', 'B2_p50', 'B8_p50', 'B11_p50', 'B12_p50',
    'B4_p75', 'B3_p75', 'B2_p75', 'B8_p75', 'B11_p75', 'B12_p75'
]

temporal_metrics = temporal_metrics.select(band_order).rename(renamed_bands)

# ===============================
# 7. Visualization
# ===============================

Map = geemap.Map()
Map.centerObject(study_area, zoom=10)

# Study area
Map.addLayer(study_area, {'color': 'FF0000'}, 'Study Area')

# EnMap RGB visualization
Map.addLayer(
    enmap_image,
    {'bands': ['b40', 'b30', 'b20'], 'min': 0, 'max': 3000},
    'EnMap RGB'
)

# Sentinel-2 temporal metrics (example visualization)
Map.addLayer(
    temporal_metrics.select(['Red_p25', 'Green_p25', 'Blue_p25']),
    {'min': 0, 'max': 3000},
    'Sentinel-2 p25 RGB'
)

Map.addLayerControl()
Map

# ===============================
# 8. Export temporal metrics to GEE Asset
# ===============================

export_task = ee.batch.Export.image.toAsset(
    image=temporal_metrics,
    description=EXPORT_DESCRIPTION,
    assetId=EXPORT_ASSET_ID,
    scale=EXPORT_SCALE,
    region=study_area,
    maxPixels=1e13
)

export_task.start()

print('Export task started successfully.')
