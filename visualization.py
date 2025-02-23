import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd

class SupplyChainVisualizer:
    @staticmethod
    def create_demand_forecast_plot(historical: List[float], forecast: List[float]) -> dict:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=historical, name="Historical"))
        fig.add_trace(go.Scatter(y=forecast, name="Forecast"))
        return fig.to_dict()
    
    @staticmethod
    def create_inventory_heatmap(inventory_levels: Dict[str, Dict[str, float]]) -> dict:
        df = pd.DataFrame(inventory_levels)
        fig = px.imshow(df, aspect="auto")
        return fig.to_dict()
    
    @staticmethod
    def create_risk_dashboard(risk_metrics: Dict[str, float]) -> dict:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_metrics["overall_risk"],
            gauge={"axis": {"range": [0, 100]}}
        ))
        return fig.to_dict()