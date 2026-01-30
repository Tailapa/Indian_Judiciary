# Instructions for Running Bengaluru Lake Analysis

This package contains automated tools to detect lakes in Bengaluru from satellite data and perform a multi-decade analysis (2010-2025) of their environmental health.

## Prerequisites
1. **Google Earth Engine (GEE) Account**: Ensure you have a registered account at [earthengine.google.com](https://earthengine.google.com/).
2. **Project ID**: The script is configured to use your project ID: `bengaluru-lakes-485612`.
3. **Python Environment**: Install required libraries:
   ```bash
   pip install earthengine-api pandas numpy scikit-learn matplotlib seaborn tqdm
   ```

## Step 1: Authentication
Before running the scripts, you must authenticate GEE on your local machine:
```bash
earthengine authenticate
```
Alternatively, you can add `ee.Authenticate()` at the start of `lake_analysis_gee.py`.

## Step 2: Data Extraction
Run the GEE extraction script. This script will:
1. Automatically detect lakes > 2 hectares within the Bengaluru bounding box using the JRC Global Surface Water dataset.
2. Create a master list of these lakes in `Bengaluru_Lakes_Master_Audit_2024.csv`.
3. Extract annual time-series data for Area, Rainfall, Green Cover, Encroachment, and Flooding.
```bash
python lake_analysis_gee.py
```

## Step 3: Analysis and Modeling
Once the extraction is complete (producing `Bengaluru_Lakes_Time_Series_2010_2025.csv`), run the modeling script:
```bash
python lake_modeling.py
```

### Outputs Generated:
- **Analysis_Summary.txt**: Key metrics and model performance summary.
- **lake_area_trends.png**: Time-series visualization of lake areas.
- **encroachment_vs_flooding_regression.png**: Regression analysis of built-up impact.
- **flood_feature_importance.png**: AI model's assessment of flood drivers.
- **metrics_correlation_heatmap.png**: Statistical correlations between all extracted features.

## Accuracy Notes
- **Area Measurement**: Uses MNDWI (Modified Normalized Difference Water Index) which is superior to NDWI in urban settings as it reduces noise from built-up shadows.
- **Encroachment**: Specifically looks for built-up spectral signatures (NDBI) *inside* the historical water occurrence boundaries of each lake.
- **Flooding**: Detects water pixels in areas that historically (JRC data) have < 10% water occurrence.
