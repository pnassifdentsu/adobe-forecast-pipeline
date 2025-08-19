import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import warnings

class DataProcessor:
    """Data preprocessing pipeline for SEM forecasting data"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from Excel file"""
        try:
            # Try reading the Excel file with different engines
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except PermissionError:
                # If permission denied, provide helpful error message
                self.logger.error(f"Permission denied accessing {file_path}")
                self.logger.error("Possible solutions:")
                self.logger.error("1. Close the Excel file if it's open")
                self.logger.error("2. Wait for OneDrive sync to complete")
                self.logger.error("3. Copy the file to a local directory")
                raise PermissionError(f"Cannot access {file_path}. Please close Excel and try again.")
            except:
                # Try with xlrd engine as fallback
                df = pd.read_excel(file_path, engine='xlrd')
                
            self.logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise
    
    def validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns exist"""
        required_cols = ['date'] + self.config.get('target_columns', [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
            for col in missing_cols:
                if col in self.config.get('target_columns', []):
                    df[col] = 0  # Fill missing target columns with 0
        
        self.logger.info("Column validation completed")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Fill missing values in cost column (regressor)
        if 'cost' in df.columns:
            missing_cost = df['cost'].isna().sum()
            if missing_cost > 0:
                df['cost'] = df['cost'].ffill()
                self.logger.warning(f"Forward-filled {missing_cost} missing values in 'cost' column")
        
        # Fill missing values in target columns
        target_cols = self.config.get('target_columns', [])
        for col in target_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna(0)
                    self.logger.warning(f"Filled {missing_count} missing values in '{col}' with 0")
        
        self.logger.info(f"Data preprocessing completed: {df.shape[0]} rows")
        return df
    
    def filter_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter dataframe by date range"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        filtered_df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
        self.logger.info(f"Filtered to date range {start_date} to {end_date}: {filtered_df.shape[0]} rows")
        
        return filtered_df
    
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
    
    def aggregate_to_granularity(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Aggregate data to specified granularity"""
        df = df.copy()
        
        if granularity == "daily":
            return df
        elif granularity == "weekly":
            # Week starts on Monday
            df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')
            agg_df = df.groupby('week_start').agg({
                col: 'sum' for col in self.config.get('target_columns', []) if col in df.columns
            }).reset_index()
            agg_df = agg_df.rename(columns={'week_start': 'date'})
            
            # Add cost as mean for weekly aggregation
            if 'cost' in df.columns:
                cost_agg = df.groupby('week_start')['cost'].mean().reset_index()
                agg_df = agg_df.merge(cost_agg.rename(columns={'week_start': 'date'}), on='date')
                
        elif granularity == "monthly":
            df['month_start'] = df['date'].dt.to_period('M').dt.start_time
            agg_df = df.groupby('month_start').agg({
                col: 'sum' for col in self.config.get('target_columns', []) if col in df.columns
            }).reset_index()
            agg_df = agg_df.rename(columns={'month_start': 'date'})
            
            # Add cost as mean for monthly aggregation
            if 'cost' in df.columns:
                cost_agg = df.groupby('month_start')['cost'].mean().reset_index()
                agg_df = agg_df.merge(cost_agg.rename(columns={'month_start': 'date'}), on='date')
        
        self.logger.info(f"Aggregated data to {granularity}: {agg_df.shape[0]} rows")
        return agg_df
    
    def create_complete_monthly_forecast(self, historical_df: pd.DataFrame, forecasts: Dict[str, pd.DataFrame], 
                                       train_end_date: str) -> Dict[str, pd.DataFrame]:
        """Create complete monthly forecasts by combining partial historical data with daily forecasts"""
        train_end = pd.to_datetime(train_end_date)
        complete_monthly_forecasts = {}
        
        # Check if training ends mid-month
        if train_end.day < 28:  # Not end of month (accounting for Feb)
            self.logger.info(f"Training ends mid-month ({train_end.strftime('%Y-%m-%d')}), creating complete monthly forecasts")
            
            # Get partial month data from historical
            partial_month_start = train_end.replace(day=1)
            partial_historical = historical_df[
                (historical_df['date'] >= partial_month_start) & 
                (historical_df['date'] <= train_end)
            ].copy()
            
            if not partial_historical.empty:
                self.logger.info(f"Found partial month data from {partial_month_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
                
                for variable, daily_forecast in forecasts.items():
                    if variable in partial_historical.columns and not daily_forecast.empty:
                        # Get forecast data for the rest of the month
                        month_end = (partial_month_start + pd.offsets.MonthEnd(0)).normalize()
                        forecast_for_month = daily_forecast[
                            (daily_forecast['ds'] > train_end) & 
                            (daily_forecast['ds'] <= month_end)
                        ].copy()
                        
                        if not forecast_for_month.empty:
                            # Calculate partial historical sum
                            partial_sum = partial_historical[variable].sum()
                            
                            # Calculate forecasted sum for rest of month
                            forecast_sum = forecast_for_month['yhat'].sum()
                            forecast_sum_lower = forecast_for_month['yhat_lower'].sum()
                            forecast_sum_upper = forecast_for_month['yhat_upper'].sum()
                            
                            # Create complete month forecast
                            complete_forecast = pd.DataFrame({
                                'ds': [partial_month_start],
                                'yhat': [partial_sum + forecast_sum],
                                'yhat_lower': [partial_sum + forecast_sum_lower],
                                'yhat_upper': [partial_sum + forecast_sum_upper]
                            })
                            
                            # Add additional months if forecast extends beyond
                            remaining_forecast = daily_forecast[daily_forecast['ds'] > month_end].copy()
                            if not remaining_forecast.empty:
                                # Group remaining forecast by month
                                remaining_forecast['month_start'] = remaining_forecast['ds'].dt.to_period('M').dt.start_time
                                monthly_remaining = remaining_forecast.groupby('month_start').agg({
                                    'yhat': 'sum',
                                    'yhat_lower': 'sum', 
                                    'yhat_upper': 'sum'
                                }).reset_index()
                                monthly_remaining = monthly_remaining.rename(columns={'month_start': 'ds'})
                                
                                # Combine complete current month with remaining months
                                complete_forecast = pd.concat([complete_forecast, monthly_remaining], ignore_index=True)
                            
                            complete_monthly_forecasts[variable] = complete_forecast
                            
                            self.logger.info(f"Created complete monthly forecast for {variable}: "
                                           f"partial historical ({partial_sum:.1f}) + forecast ({forecast_sum:.1f}) = "
                                           f"total ({partial_sum + forecast_sum:.1f})")
            
        else:
            # Training ends at month end, just aggregate daily forecasts normally
            for variable, daily_forecast in forecasts.items():
                if not daily_forecast.empty:
                    # Group by month
                    monthly_forecast = daily_forecast.copy()
                    monthly_forecast['month_start'] = monthly_forecast['ds'].dt.to_period('M').dt.start_time
                    monthly_agg = monthly_forecast.groupby('month_start').agg({
                        'yhat': 'sum',
                        'yhat_lower': 'sum',
                        'yhat_upper': 'sum'
                    }).reset_index()
                    monthly_agg = monthly_agg.rename(columns={'month_start': 'ds'})
                    complete_monthly_forecasts[variable] = monthly_agg
        
        self.logger.info(f"Generated complete monthly forecasts for {len(complete_monthly_forecasts)} variables")
        return complete_monthly_forecasts
    
    def validate_results(self, df: pd.DataFrame, step_name: str):
        """Validate results after each major step"""
        if df.empty:
            self.logger.error(f"{step_name}: DataFrame is empty")
            raise ValueError(f"Empty DataFrame after {step_name}")
        
        if df.isna().all().any():
            self.logger.warning(f"{step_name}: Some columns are entirely NaN")
        
        self.logger.info(f"{step_name}: Validation passed - {df.shape[0]} rows, {df.shape[1]} columns")