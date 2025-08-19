import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ProphetForecaster:
    """Prophet-based forecasting model with external regressors"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        
    def create_future_dataframe(self, model: Prophet, periods: int, freq: str, 
                              last_known_values: Dict) -> pd.DataFrame:
        """Create future dataframe with imputed regressor values"""
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Add all required regressor columns
        regressor_cols = ['cost', 'day_of_year', 'month', 'quarter']
        for col in regressor_cols:
            if col == 'cost':
                # For cost, forward fill from training data, then use last known value for future
                if col in future.columns:
                    future[col] = future[col].ffill()
                    if future[col].isna().any():
                        future[col] = future[col].fillna(last_known_values.get(col, 0))
                else:
                    # If cost column doesn't exist, add it with last known value
                    future[col] = last_known_values.get(col, 0)
                
                missing_count = future[col].isna().sum()
                if missing_count > 0:
                    self.logger.warning(f"Imputed {missing_count} missing '{col}' values")
                    
            else:
                # For date-based features, always recalculate from ds column
                if col == 'day_of_year':
                    future[col] = future['ds'].dt.dayofyear
                elif col == 'month':
                    future[col] = future['ds'].dt.month
                elif col == 'quarter':
                    future[col] = future['ds'].dt.quarter
        
        return future
    
    def fit_model(self, train_data: pd.DataFrame, target_column: str) -> Prophet:
        """Fit Prophet model for a specific target variable"""
        try:
            # Initialize Prophet with configuration
            prophet_params = self.config.get('prophet_params', {})
            model = Prophet(
                yearly_seasonality=prophet_params.get('yearly_seasonality', True),
                weekly_seasonality=prophet_params.get('weekly_seasonality', True),
                daily_seasonality=prophet_params.get('daily_seasonality', False),
                seasonality_mode=prophet_params.get('seasonality_mode', 'multiplicative'),
                changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10),
                interval_width=prophet_params.get('interval_width', 0.95)
            )
            
            # Add external regressors
            if 'cost' in train_data.columns:
                model.add_regressor('cost')
            
            # Add date-based regressors
            date_regressors = ['day_of_year', 'month', 'quarter']
            for regressor in date_regressors:
                if regressor in train_data.columns:
                    model.add_regressor(regressor)
            
            # Fit model
            model.fit(train_data)
            self.models[target_column] = model
            
            self.logger.info(f"Prophet model fitted for {target_column}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to fit model for {target_column}: {str(e)}")
            raise
    
    def generate_forecast(self, model: Prophet, train_data: pd.DataFrame, 
                         forecast_start: str, forecast_end: str, 
                         granularity: str) -> pd.DataFrame:
        """Generate forecast for specified period"""
        try:
            # Calculate periods to forecast
            start_date = pd.to_datetime(forecast_start)
            end_date = pd.to_datetime(forecast_end)
            
            if granularity == "daily":
                periods = (end_date - start_date).days + 1
                freq = 'D'
            elif granularity == "weekly":
                periods = int(np.ceil((end_date - start_date).days / 7)) + 1
                freq = 'W-MON'  # Weekly starting Monday
            elif granularity == "monthly":
                periods = ((end_date.year - start_date.year) * 12 + 
                          end_date.month - start_date.month) + 1
                # Ensure at least 1 period for monthly forecasts
                periods = max(periods, 1)
                freq = 'MS'  # Month start
            
            # Get last known values for imputation
            last_known_values = {}
            if 'cost' in train_data.columns:
                last_known_values['cost'] = train_data['cost'].iloc[-1]
            
            # Create future dataframe
            future = self.create_future_dataframe(model, periods, freq, last_known_values)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Filter to forecast period only
            forecast_only = forecast[forecast['ds'] >= start_date].copy()
            forecast_only = forecast_only[forecast_only['ds'] <= end_date].copy()
            
            self.logger.info(f"Forecast generated: {len(forecast_only)} periods")
            return forecast_only
            
        except Exception as e:
            self.logger.error(f"Failed to generate forecast: {str(e)}")
            raise
    
    def forecast_all_variables(self, train_data: pd.DataFrame, granularity: str) -> Dict[str, pd.DataFrame]:
        """Generate forecasts for all target variables"""
        forecasts = {}
        target_columns = self.config.get('target_columns', [])
        
        for target_col in target_columns:
            if target_col in train_data.columns:
                try:
                    # Prepare data for Prophet
                    prophet_data = self.prepare_prophet_data(train_data, target_col)
                    
                    # Fit model
                    model = self.fit_model(prophet_data, target_col)
                    
                    # Generate forecast
                    forecast = self.generate_forecast(
                        model, prophet_data,
                        self.config.get('forecast_start'),
                        self.config.get('forecast_end'),
                        granularity
                    )
                    
                    forecasts[target_col] = forecast
                    
                except Exception as e:
                    self.logger.error(f"Failed to forecast {target_col}: {str(e)}")
                    continue
        
        self.logger.info(f"Generated forecasts for {len(forecasts)} variables")
        return forecasts
    
    def prepare_prophet_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df['date']
        prophet_df['y'] = df[target_column]
        
        # Add regressors
        if 'cost' in df.columns:
            prophet_df['cost'] = df['cost']
        
        # Add date-based features as regressors
        prophet_df['day_of_year'] = prophet_df['ds'].dt.dayofyear
        prophet_df['month'] = prophet_df['ds'].dt.month
        prophet_df['quarter'] = prophet_df['ds'].dt.quarter
        
        return prophet_df
    
    def validate_forecast(self, forecast: pd.DataFrame, variable: str, granularity: str):
        """Validate forecast results"""
        if forecast.empty:
            self.logger.error(f"Empty forecast for {variable} at {granularity}")
            raise ValueError(f"Empty forecast generated")
        
        required_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        missing_cols = [col for col in required_cols if col not in forecast.columns]
        if missing_cols:
            self.logger.error(f"Missing forecast columns: {missing_cols}")
            raise ValueError(f"Missing columns in forecast: {missing_cols}")
        
        # Check for infinite or extreme values
        if forecast[['yhat', 'yhat_lower', 'yhat_upper']].isin([np.inf, -np.inf]).any().any():
            self.logger.warning(f"Infinite values detected in forecast for {variable}")
        
        self.logger.info(f"Forecast validation passed for {variable} ({granularity}): {len(forecast)} periods")