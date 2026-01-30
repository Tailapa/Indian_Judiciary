import ee
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os

# 1. Initialize Google Earth Engine
# Authentication is required for first-time use: ee.Authenticate()
try:
    ee.Initialize(project='bengaluru-lakes-485612')
    print("GEE Initialized successfully.")
except Exception as e:
    print(f"GEE Initialization failed: {e}")
    print("Please ensure you have authenticated.")

def detect_and_analyze_lakes(start_year=2010, end_year=2025):
    # Bengaluru Bounding Box
    roi = ee.Geometry.Rectangle([77.46, 12.83, 77.78, 13.14])

    # 1. LAKE DETECTION (Automated)
    # Use JRC Global Surface Water occurrence to find permanent/semi-permanent water bodies
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    # Define lakes as areas with water occurrence > 50%
    water_mask = gsw.select('occurrence').gt(50).clip(roi)

    # Connected Components to identify distinct lakes
    objects = water_mask.selfMask().connectedComponents(
        connectedness=ee.Kernel.plus(1),
        maxSize=1000
    )

    # Calculate area and filter small ponds (e.g., < 2 hectares)
    area_image = ee.Image.pixelArea().addBands(objects.select('labels').toInt())
    object_area = area_image.reduceConnectedComponents(
        reducer=ee.Reducer.sum(),
        labelBand='labels'
    )

    large_lakes_mask = object_area.gt(20000)

    # Get centroids of detected lakes
    lake_vectors = water_mask.updateMask(large_lakes_mask).reduceToVectors(
        geometry=roi,
        scale=30,
        geometryType='centroid',
        eightConnected=True,
        labelProperty='lake_id',
        reducer=ee.Reducer.countEvery()
    )

    lake_list = lake_vectors.getInfo()['features']
    print(f"Detected {len(lake_list)} lakes in Bengaluru.")

    # Create the Master CSV if needed
    master_data = []
    for i, feature in enumerate(lake_list):
        coords = feature['geometry']['coordinates']
        master_data.append({
            'name': f"Detected_Lake_{i+1}",
            'lat': coords[1],
            'lon': coords[0]
        })
    pd.DataFrame(master_data).to_csv('Bengaluru_Lakes_Master_Audit_2024.csv', index=False)

    # 2. TIME SERIES EXTRACTION
    all_results = []

    for year in range(start_year, end_year + 1):
        print(f"Processing year: {year}")
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'

        # Consistent Band Mapping functions
        def rename_l89(img):
            return img.select(
                ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'],
                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'QA_PIXEL']
            )

        def rename_l57(img):
            return img.select(
                ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'],
                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'QA_PIXEL']
            )

        def preprocess_landsat(img):
            # Scale factors for Landsat C2 L2
            # SR_BANDS: output = input * 0.0000275 - 0.2
            qa = img.select('QA_PIXEL')
            mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))

            # Apply scaling
            optical_bands = img.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).multiply(0.0000275).add(-0.2)
            return img.addBands(optical_bands, None, True).updateMask(mask)

        # Load Collections
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(year_start, year_end).filterBounds(roi).map(rename_l89)
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(year_start, year_end).filterBounds(roi).map(rename_l89)
        l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterDate(year_start, year_end).filterBounds(roi).map(rename_l57)
        l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterDate(year_start, year_end).filterBounds(roi).map(rename_l57)

        composite = l9.merge(l8).merge(l7).merge(l5).map(preprocess_landsat).median().clip(roi)

        # Calculate Indices using standardized bands
        mndwi = composite.normalizedDifference(['green', 'swir1']).rename('MNDWI')
        ndvi = composite.normalizedDifference(['nir', 'red']).rename('NDVI')
        ndbi = composite.normalizedDifference(['swir1', 'nir']).rename('NDBI')
        indexed = composite.addBands([mndwi, ndvi, ndbi])

        # Rainfall (CHIRPS)
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(year_start, year_end).filterBounds(roi).sum().clip(roi)

        def get_stats(lake_feature):
            lake_id = lake_feature.get('lake_id')
            lake_geom = lake_feature.geometry().buffer(300)

            water_area = indexed.select('MNDWI').gt(0).multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=lake_geom, scale=30).values().get(0)

            rain = chirps.reduceRegion(reducer=ee.Reducer.mean(), geometry=lake_geom, scale=5000).values().get(0)

            green = indexed.select('NDVI').gt(0.3).multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=lake_geom, scale=30).values().get(0)

            enc = indexed.select('NDBI').gt(0).And(water_mask).multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=lake_geom, scale=30).values().get(0)

            low_occ = gsw.select('occurrence').lt(10).And(gsw.select('occurrence').gt(0))
            flood = indexed.select('MNDWI').gt(0).And(low_occ).multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=lake_geom, scale=30).values().get(0)

            return ee.Feature(None, {
                'Lake_ID': lake_id,
                'Year': year,
                'Lake_Area_sqm': water_area,
                'Rainfall_mm': rain,
                'Green_Cover_sqm': green,
                'Encroachment_sqm': enc,
                'Flood_Area_sqm': flood
            })

        stats_fc = lake_vectors.map(get_stats).getInfo()['features']
        for feat in stats_fc:
            props = feat['properties']
            props['Lake_Name'] = f"Lake_{props['Lake_ID']}"
            all_results.append(props)

    return all_results

if __name__ == "__main__":
    print("🚀 Automated Lake Detection and Analysis starting...")
    try:
        results = detect_and_analyze_lakes()
        pd.DataFrame(results).to_csv('Bengaluru_Lakes_Time_Series_2010_2025.csv', index=False)
        print("✅ Complete dataset saved to 'Bengaluru_Lakes_Time_Series_2010_2025.csv'")
    except Exception as e:
        print(f"❌ Error during GEE execution: {e}")
