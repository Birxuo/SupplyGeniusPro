import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

class SupplyChainVisualizer:
    @staticmethod
    def create_demand_forecast_plot(historical: List[float], forecast: List[float]) -> dict:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=historical, name="Historical"))
        fig.add_trace(go.Scatter(y=forecast, name="Forecast"))
        return fig.to_dict()
    
    @staticmethod
    def create_risk_heatmap(risk_data: Dict[str, float]) -> dict:
        fig = px.imshow(pd.DataFrame(risk_data).corr())
        return fig.to_dict()