from typing import List, Dict
import asyncio
from datetime import datetime

class AlertSystem:
    def __init__(self):
        self.alert_thresholds = {
            "inventory": 0.2,
            "demand_spike": 0.3,
            "supplier_delay": 48,
            "quality_issues": 0.05
        }

    async def check_inventory_alerts(self, inventory_levels: Dict[str, float]) -> List[dict]:
        alerts = []
        for item, level in inventory_levels.items():
            if level < self.alert_thresholds["inventory"]:
                alerts.append({
                    "type": "inventory",
                    "severity": "high",
                    "item": item,
                    "current_level": level,
                    "timestamp": datetime.now().isoformat()
                })
        return alerts
