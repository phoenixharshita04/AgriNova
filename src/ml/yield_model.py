import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

class YieldPredictor:
    def __init__(self):
        # Base model for GridSearch
        self.base_model = RandomForestRegressor(random_state=42)
        # Optimized model will be stored here
        self.model = None
        self.features = ['Avg_Temp', 'Total_Rainfall', 'Avg_Soil_Moisture', 'Max_NDVI', 'Avg_NDVI', 'Interaction_Temp_Moisture']
        self.target = 'Yield_tons_per_ha'
        
    def _engineer_features(self, df):
        """
        Adds advanced interaction and time-series features.
        """
        df_new = df.copy()
        # Interaction feature: Temperature multiplied by Soil Moisture
        if 'Interaction_Temp_Moisture' not in df_new.columns:
            df_new['Interaction_Temp_Moisture'] = df_new['Avg_Temp'] * df_new['Avg_Soil_Moisture']
            
        # If this is historical time-series data with a Date column, we could add rolling averages
        # For simplicity in this aggregated seasonal dataset, we rely on the interaction term.
        return df_new
        
    def train_strict_split(self, df):
        """
        Trains the model using TimeSeriesSplit and GridSearchCV for hyperparameter optimization.
        """
        # Ensure data is sorted by time (Year in this case)
        df_sorted = df.sort_values(by='Year').reset_index(drop=True)
        df_sorted = self._engineer_features(df_sorted)
        
        X = df_sorted[self.features]
        y = df_sorted[self.target]
        
        # Strict time-series split
        tscv = TimeSeriesSplit(n_splits=max(2, min(3, len(df_sorted)-2)))
        
        # Hyperparameter Grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        
        print("Starting GridSearchCV for Yield Model with Time-Series Cross Validation...")
        grid_search = GridSearchCV(
            estimator=self.base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        
        print(f"Best Hyperparameters: {grid_search.best_params_}")
        best_rmse = np.sqrt(-grid_search.best_score_)
        print(f"Best CV RMSE: {best_rmse:.4f}")
        print("Final optimized model trained on all historical data.")
        
    def predict(self, current_features):
        """
        Predicts yield based on current season features.
        """
        df_features = pd.DataFrame([current_features])
        df_features = self._engineer_features(df_features)
        
        # Ensure column order matches
        X = df_features[self.features]
        return self.model.predict(X)[0]
        
    def predict_what_if(self, current_features, scenario):
        """
        Modifies the current features based on the climate scenario and predicts yield.
        """
        stressed_features = current_features.copy()
        
        if scenario == 'Severe Drought':
            stressed_features['Total_Rainfall'] *= 0.3
            stressed_features['Avg_Soil_Moisture'] *= 0.5
            stressed_features['Avg_NDVI'] *= 0.8
        elif scenario == 'Unseasonal Heatwave':
            stressed_features['Avg_Temp'] += 5.0
            stressed_features['Avg_Soil_Moisture'] *= 0.7
            stressed_features['Max_NDVI'] *= 0.9
        # 'Normal Monsoon' or baseline implies no changes
            
        df_features = pd.DataFrame([stressed_features])
        df_features = self._engineer_features(df_features)
        X = df_features[self.features]
        return self.model.predict(X)[0]
        
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Helper function to generate a dummy model for immediate dashboard usage
def get_dummy_yield_model():
    predictor = YieldPredictor()
    # Create some dummy data to fit the model so it can predict (scaled for Indian crops)
    dummy_data = pd.DataFrame({
        'Year': [2018, 2019, 2020, 2021, 2022],
        'Avg_Temp': [31, 33, 30, 32, 31.5],
        'Total_Rainfall': [750, 900, 600, 1100, 850],
        'Avg_Soil_Moisture': [40, 45, 38, 50, 42],
        'Max_NDVI': [0.75, 0.8, 0.65, 0.85, 0.75],
        'Avg_NDVI': [0.55, 0.6, 0.45, 0.65, 0.55],
        'Yield_tons_per_ha': [3.5, 4.2, 2.8, 4.8, 3.9] # Roughly scaled for generic cereal crops
    })
    predictor.train_strict_split(dummy_data)
    return predictor
