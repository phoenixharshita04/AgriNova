import pandas as pd
import numpy as np

def simulate_ndvi_data(start_date, end_date, region="Punjab"):
    """
    Simulates NDVI (Normalized Difference Vegetation Index) time-series data.
    NDVI typically ranges from -1 to 1. For healthy crops, it's usually between 0.3 and 0.8.
    It follows a seasonal pattern: low at planting, peaks at maturity, drops at harvest.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a base seasonal curve using sine wave
    days = np.arange(len(dates))
    season_length = 120 # typical crop season length
    
    # Simulate a crop growth cycle (NDVI increases then decreases)
    # Adding some noise
    base_ndvi = 0.2 + 0.6 * np.sin(np.pi * (days % season_length) / season_length)
    base_ndvi = np.clip(base_ndvi, 0.1, 0.9) # Keep within realistic bounds
    
    noise = np.random.normal(0, 0.05, len(dates))
    ndvi_values = np.clip(base_ndvi + noise, 0.0, 1.0)
    
    df = pd.DataFrame({
        'Date': dates,
        'Region': region,
        'NDVI': ndvi_values
    })
    return df

def fetch_sentinel2_data(region, date):
    """
    Placeholder for actual Sentinel-2 API integration.
    Returns a dummy NDVI value for a specific date and region.
    """
    # Simulate a recent NDVI value
    return round(np.random.uniform(0.4, 0.8), 2)
