import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

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
        # Simplified for demo
        return {
            "rf_forecast": [0.0] * forecast_horizon,
            "gb_forecast": [0.0] * forecast_horizon,
            "ensemble_forecast": [0.0] * forecast_horizon
        }

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['trend'] = np.arange(len(df))
        df['season'] = np.sin(2 * np.pi * df.index / 365.25)
        return df

    def _create_sequences(self, data: np.ndarray, window_size: int = 12) -> tuple:
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
