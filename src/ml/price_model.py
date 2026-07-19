import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import pickle

class PriceForecaster:
    def __init__(self, lag_days=7):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lag_days = lag_days
        
    def _create_lag_features(self, df):
        """Creates lag features for time-series forecasting."""
        data = df.copy()
        for i in range(1, self.lag_days + 1):
            data[f'Lag_{i}'] = data['Price_INR_per_Qtl'].shift(i)
        return data.dropna()
        
    def train(self, df):
        """
        Trains the forecaster on the historical price data.
        """
        data = self._create_lag_features(df)
        features = [col for col in data.columns if col.startswith('Lag_')]
        
        X = data[features]
        y = data['Price_INR_per_Qtl']
        
        self.model.fit(X, y)
        # Store the last sequence for future prediction
        self.last_sequence = data['Price_INR_per_Qtl'].iloc[-self.lag_days:].values
        
    def forecast(self, days=30):
        """
        Forecasts prices for the next `days` days.
        """
        predictions = []
        current_seq = list(self.last_sequence)
        
        for _ in range(days):
            # Features must match the lag order: Lag_1 is the most recent (last item in seq), Lag_N is the oldest (first item)
            # So if sequence is [day-7, day-6, ..., day-1]
            # Lag_1 = day-1, Lag_2 = day-2... Lag_7 = day-7
            # We reverse the sequence to match [Lag_1, Lag_2... Lag_N]
            features_array = np.array(current_seq[::-1]).reshape(1, -1)
            feature_names = [f'Lag_{i}' for i in range(1, self.lag_days + 1)]
            features_df = pd.DataFrame(features_array, columns=feature_names)
            pred = self.model.predict(features_df)[0]
            predictions.append(pred)
            
            # Update sequence (drop oldest, append newest pred)
            current_seq.pop(0)
            current_seq.append(pred)
            
        return predictions

def generate_price_recommendation(forecast_prices):
    """
    Analyzes the forecast to provide a 'Sell Now' or 'Hold' recommendation.
    If the average of the last 7 days of the forecast is significantly higher 
    than the first 7 days, it recommends 'Hold'. Otherwise 'Sell Now'.
    """
    if len(forecast_prices) < 30:
        return "Not enough forecast data", "neutral"
        
    start_avg = np.mean(forecast_prices[:7])
    end_avg = np.mean(forecast_prices[-7:])
    
    # 2% growth threshold to hold
    if end_avg > start_avg * 1.02:
        return "Hold", "Prices are projected to rise over the next month. Delay selling for higher margins."
    else:
        return "Sell Now", "Prices are expected to plateau or drop. It is advisable to liquidate inventory soon."

# Singleton loader for the dashboard
def get_price_model(historical_df):
    forecaster = PriceForecaster(lag_days=14)
    forecaster.train(historical_df)
    return forecaster
