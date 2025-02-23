import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf

class SupplyChainMLPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=100)
        self.gb_model = GradientBoostingRegressor()
        
    def preprocess_time_series(self, data: Dict[str, List[float]]) -> np.ndarray:
        df = pd.DataFrame(data)
        df = self._add_temporal_features(df)
        return self.scaler.fit_transform(df)
    
    def generate_ml_forecasts(self, data: np.ndarray, forecast_horizon: int = 12) -> Dict[str, List[float]]:
        X, y = self._create_sequences(data)
        self.rf_model.fit(X, y)
        self.gb_model.fit(X, y)
        
        rf_forecast = self.rf_model.predict(X[-forecast_horizon:])
        gb_forecast = self.gb_model.predict(X[-forecast_horizon:])
        
        return {
            "rf_forecast": rf_forecast.tolist(),
            "gb_forecast": gb_forecast.tolist(),
            "ensemble_forecast": ((rf_forecast + gb_forecast) / 2).tolist()
        }
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['trend'] = np.arange(len(df))
        df['season'] = np.sin(2 * np.pi * df.index / 365.25)
        return df
    
    def _create_sequences(self, data: np.ndarray, window_size: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)