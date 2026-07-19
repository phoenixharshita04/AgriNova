import pandas as pd
import numpy as np
from src.data.remote_sensing import simulate_ndvi_data

def generate_synthetic_features(start_date, end_date, region="Punjab"):
    """
    Generates synthetic time-series data for weather, soil, and NDVI simulating Indian climate.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Simulate Indian Weather Data (Monsoon effect based on month)
    months = dates.month
    # Summer/Monsoon (June-Sept) gets heavy rain, rest is dry
    is_monsoon = (months >= 6) & (months <= 9)
    
    temp_c = np.random.normal(32, 6, n_days) # Avg 32C, std 6
    temp_c = np.where(months > 10, temp_c - 10, temp_c) # cooler winters
    
    rainfall_mm = np.where(is_monsoon, np.random.exponential(25, n_days), np.random.exponential(2, n_days))
    humidity = np.clip(np.where(is_monsoon, np.random.normal(85, 10, n_days), np.random.normal(50, 15, n_days)), 0, 100)
    
    # Simulate Soil Data
    soil_moisture = np.clip(np.random.normal(35, 10, n_days) + (rainfall_mm * 0.8), 0, 100)
    
    df = pd.DataFrame({
        'Date': dates,
        'Region': region,
        'Temperature_C': temp_c,
        'Rainfall_mm': rainfall_mm,
        'Humidity_pct': humidity,
        'Soil_Moisture_pct': soil_moisture
    })
    
    # Get NDVI data
    ndvi_df = simulate_ndvi_data(start_date, end_date, region)
    
    # Merge datasets
    merged_df = pd.merge(df, ndvi_df, on=['Date', 'Region'])
    
    # Engineer some time-series features
    # 7-day rolling averages
    merged_df['Temp_7d_avg'] = merged_df['Temperature_C'].rolling(window=7, min_periods=1).mean()
    merged_df['Rainfall_7d_sum'] = merged_df['Rainfall_mm'].rolling(window=7, min_periods=1).sum()
    merged_df['NDVI_7d_avg'] = merged_df['NDVI'].rolling(window=7, min_periods=1).mean()
    
    return merged_df

def prepare_yield_dataset(years=5, region="Punjab"):
    """
    Creates a dataset for yield prediction by aggregating seasonal data.
    """
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    
    df = generate_synthetic_features(start_date, end_date, region)
    
    # Aggregate data by "season" (e.g., assuming 1 season per year for simplicity)
    df['Year'] = df['Date'].dt.year
    
    seasonal_data = []
    for year, group in df.groupby('Year'):
        if len(group) < 100: continue # Skip incomplete years
        
        season_summary = {
            'Year': year,
            'Region': region,
            'Avg_Temp': group['Temperature_C'].mean(),
            'Total_Rainfall': group['Rainfall_mm'].sum(),
            'Avg_Soil_Moisture': group['Soil_Moisture_pct'].mean(),
            'Max_NDVI': group['NDVI'].max(),
            'Avg_NDVI': group['NDVI'].mean()
        }
        
        # Simulate Yield based on these features (simple linear relation + noise)
        # Higher rainfall and NDVI generally mean higher yield (up to a point)
        base_yield = 2.0 # baseline tons per hectare
        yield_val = base_yield + (season_summary['Total_Rainfall'] * 0.001) + (season_summary['Max_NDVI'] * 1.5)
        yield_val += np.random.normal(0, 0.2) # Add noise
        
        season_summary['Yield_tons_per_ha'] = max(0.5, yield_val)
        
        seasonal_data.append(season_summary)
        
        
    return pd.DataFrame(seasonal_data), df

def generate_mandi_prices(crop_type, start_date, end_date):
    """
    Generates simulated historical Mandi prices (INR/Quintal) for the specified crop.
    Includes baseline price, seasonal trends (harvest drop), and noise.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Baseline prices roughly based on Indian MSP / Market rates (INR/Quintal)
    baselines = {
        "Rice": 2200, "Wheat": 2300, "Cotton": 6500, 
        "Sugarcane": 350, "Soybean": 4600, "Tomato": 1500, "Potato": 1200
    }
    base_price = baselines.get(crop_type, 2000)
    
    # Simulate a yearly seasonal cycle (prices drop during harvest season)
    days = np.arange(n_days)
    seasonality = np.sin(2 * np.pi * days / 365.25) * (base_price * 0.15)
    
    # Long-term drift (inflation/market trend)
    trend = np.linspace(0, base_price * 0.1, n_days)
    
    # Random daily volatility
    noise = np.random.normal(0, base_price * 0.05, n_days)
    
    prices = np.clip(base_price + seasonality + trend + noise, base_price * 0.5, base_price * 2.0)
    
    df = pd.DataFrame({
        'Date': dates,
        'Crop': crop_type,
        'Price_INR_per_Qtl': prices
    })
    return df
    
# End of feature_engineering.py
