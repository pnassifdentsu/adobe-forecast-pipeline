from datetime import datetime
from typing import Dict, List, Any
import json

class ForecastConfig:
    """Configuration class for the Prophet forecasting pipeline"""
    
    def __init__(self, config_path: str = None):
        self.default_config = {
            "data_file": "Data/adobe_forecast_data_local.xlsx",
            "train_start": "2023-01-01",
            "train_end": "2025-08-16", 
            "forecast_start": "2025-08-17",
            "forecast_end": "2025-09-30",
            "output_dir": "output",
            "granularities": ["daily", "weekly", "monthly"],
            "target_columns": [
                "Visits", "NC Visits", "CC Visits", "Mobile Visits",
                "Orders", "NCOs", "CCOs", "Mobile Orders", 
                "Call Volume", "Call Orders", "NCO Call Orders", "Mobile Call Orders"
            ],
            "regressor_columns": ["date", "cost"],
            "prophet_params": {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "seasonality_mode": "multiplicative",
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10,
                "interval_width": 0.95
            }
        }
        
        if config_path:
            self.load_config(config_path)
        else:
            self.config = self.default_config.copy()
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        self.config = {**self.default_config, **user_config}
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value