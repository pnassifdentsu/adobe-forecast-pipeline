import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

class OutputManager:
    """Handles JSON output and PNG visualization generation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = config.get('output_dir', 'output')
        self.run_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def format_forecast_to_json(self, forecasts: Dict[str, pd.DataFrame], 
                               granularity: str) -> List[Dict]:
        """Format forecast results to JSON schema"""
        json_records = []
        
        for variable, forecast_df in forecasts.items():
            for _, row in forecast_df.iterrows():
                record = {
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "granularity": granularity,
                    "variable": variable,
                    "forecast": float(row['yhat']),
                    "lower_bound": float(row['yhat_lower']),
                    "upper_bound": float(row['yhat_upper'])
                }
                json_records.append(record)
        
        # Sort by date, then by variable for consistent output
        json_records.sort(key=lambda x: (x['date'], x['variable']))
        
        self.logger.info(f"Formatted {len(json_records)} forecast records for {granularity}")
        return json_records
    
    def save_json_output(self, forecast_data: List[Dict], granularity: str) -> str:
        """Save forecast data to JSON file"""
        filename = f"prophet_forecast_{granularity}_{self.run_date}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(forecast_data, f, indent=2)
            
            self.logger.info(f"JSON output saved: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON output: {str(e)}")
            raise
    
    def filter_historical_data_for_viz(self, historical_df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Filter historical data to start from 2024-01-01"""
        if historical_df.empty:
            return historical_df
            
        start_date = pd.to_datetime('2024-01-01')
        filtered_df = historical_df[historical_df['date'] >= start_date].copy()
        
        self.logger.info(f"Filtered historical data for visualization: {len(filtered_df)} rows from 2024-01-01")
        return filtered_df

    def create_single_pdf_visualizations(self, forecasts: Dict[str, pd.DataFrame], 
                                       historical_data: pd.DataFrame, 
                                       granularity: str) -> str:
        """Create single PDF file with all forecast visualizations for a granularity"""
        # Filter historical data to start from 2024-01-01
        viz_historical_data = self.filter_historical_data_for_viz(historical_data, granularity)
        
        filename = f"prophet_forecast_all_variables_{granularity}_{self.run_date}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        plt.style.use('seaborn-v0_8')
        
        try:
            with PdfPages(filepath) as pdf:
                # Calculate grid dimensions based on number of variables
                n_vars = len(forecasts)
                n_cols = 3  # 3 columns per page
                n_rows = 4  # 4 rows per page
                vars_per_page = n_cols * n_rows
                
                var_names = list(forecasts.keys())
                
                for page_start in range(0, n_vars, vars_per_page):
                    page_vars = var_names[page_start:page_start + vars_per_page]
                    
                    # Create figure for this page
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
                    fig.suptitle(f'Prophet Forecasts - {granularity.title()} Granularity', fontsize=20, fontweight='bold')
                    
                    # Flatten axes array for easier indexing
                    if n_rows * n_cols == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()
                    
                    for i, variable in enumerate(page_vars):
                        ax = axes[i]
                        forecast_df = forecasts[variable]
                        
                        # Plot historical data (from 2024-01-01)
                        if not viz_historical_data.empty and variable in viz_historical_data.columns:
                            ax.plot(viz_historical_data['date'], viz_historical_data[variable], 
                                   label='Historical', color='blue', linewidth=1.5, alpha=0.8)
                        
                        # For monthly granularity, handle combined historical+forecast data differently
                        if granularity == "monthly" and not forecast_df.empty:
                            # Check if first forecast date is a current/partial month (combined data)
                            first_forecast_date = forecast_df['ds'].min()
                            
                            # Plot all forecast points 
                            ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                                   label='Forecast (includes EOM projection)', color='red', linewidth=2, linestyle='--')
                            
                            # Plot confidence intervals
                            ax.fill_between(forecast_df['ds'], 
                                          forecast_df['yhat_lower'], 
                                          forecast_df['yhat_upper'], 
                                          alpha=0.3, color='red', label='95% CI')
                            
                            # Add annotation for first point if it's a combined month
                            if not viz_historical_data.empty and len(forecast_df) > 0:
                                first_point_date = forecast_df['ds'].iloc[0]
                                first_point_value = forecast_df['yhat'].iloc[0]
                                ax.annotate(f'EOM Projection\n({first_point_value:.0f})', 
                                          xy=(first_point_date, first_point_value),
                                          xytext=(10, 10), textcoords='offset points',
                                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                          arrowprops=dict(arrowstyle='->', color='black'),
                                          fontsize=8)
                        else:
                            # Normal forecast plotting for daily/weekly
                            ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                                   label='Forecast', color='red', linewidth=2, linestyle='--')
                            
                            # Plot confidence intervals
                            ax.fill_between(forecast_df['ds'], 
                                          forecast_df['yhat_lower'], 
                                          forecast_df['yhat_upper'], 
                                          alpha=0.3, color='red', label='95% CI')
                            
                            # Add vertical line to separate historical and forecast periods
                            if not forecast_df.empty:
                                forecast_start = forecast_df['ds'].min()
                                ax.axvline(x=forecast_start, color='gray', linestyle=':', alpha=0.7)
                        
                        # Formatting
                        ax.set_title(variable, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Date', fontsize=10)
                        ax.set_ylabel(variable, fontsize=10)
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)
                        
                        # Rotate x-axis labels
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        ax.tick_params(axis='y', labelsize=8)
                    
                    # Hide unused subplots
                    for i in range(len(page_vars), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            
            self.logger.info(f"PDF visualization saved: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to create PDF visualization: {str(e)}")
            raise

    def create_forecast_visualization(self, forecast_df: pd.DataFrame, 
                                    historical_df: pd.DataFrame,
                                    variable: str, granularity: str) -> str:
        """Create PNG visualization for forecast (kept for backward compatibility)"""
        # Filter historical data to start from 2024-01-01
        viz_historical_df = self.filter_historical_data_for_viz(historical_df, granularity)
        
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        try:
            # Plot historical data (from 2024-01-01)
            if not viz_historical_df.empty and variable in viz_historical_df.columns:
                ax.plot(viz_historical_df['date'], viz_historical_df[variable], 
                       label='Historical', color='blue', linewidth=2)
            
            # Plot forecast
            ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                   label='Forecast', color='red', linewidth=2, linestyle='--')
            
            # Plot confidence intervals
            ax.fill_between(forecast_df['ds'], 
                          forecast_df['yhat_lower'], 
                          forecast_df['yhat_upper'], 
                          alpha=0.3, color='red', label='95% Confidence Interval')
            
            # Formatting
            ax.set_title(f'{variable} Forecast - {granularity.title()}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(variable, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Add vertical line to separate historical and forecast periods
            if not forecast_df.empty:
                forecast_start = forecast_df['ds'].min()
                ax.axvline(x=forecast_start, color='gray', linestyle=':', alpha=0.7, 
                          label='Forecast Start')
            
            plt.tight_layout()
            
            # Save plot
            filename = f"prophet_forecast_{variable.replace(' ', '_')}_{granularity}_{self.run_date}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualization saved: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization for {variable}: {str(e)}")
            plt.close()
            raise
    
    def generate_all_visualizations(self, forecasts: Dict[str, pd.DataFrame], 
                                  historical_data: pd.DataFrame, 
                                  granularity: str) -> List[str]:
        """Generate single PDF visualization for all forecast variables"""
        visualization_files = []
        
        try:
            # Create single PDF with all variables
            pdf_filepath = self.create_single_pdf_visualizations(
                forecasts, historical_data, granularity
            )
            visualization_files.append(pdf_filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to generate PDF visualization for {granularity}: {str(e)}")
            # Fallback to individual PNG files
            self.logger.info("Falling back to individual PNG visualizations")
            for variable, forecast_df in forecasts.items():
                try:
                    filepath = self.create_forecast_visualization(
                        forecast_df, historical_data, variable, granularity
                    )
                    visualization_files.append(filepath)
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate visualization for {variable}: {str(e)}")
                    continue
        
        self.logger.info(f"Generated {len(visualization_files)} visualization files for {granularity}")
        return visualization_files
    
    def validate_json_output(self, json_data: List[Dict], granularity: str):
        """Validate JSON output format"""
        required_fields = ["date", "granularity", "variable", "forecast", "lower_bound", "upper_bound"]
        
        if not json_data:
            self.logger.error("Empty JSON output")
            raise ValueError("Empty JSON output")
        
        # Check first record structure
        first_record = json_data[0]
        missing_fields = [field for field in required_fields if field not in first_record]
        if missing_fields:
            self.logger.error(f"Missing required fields in JSON output: {missing_fields}")
            raise ValueError(f"Missing fields: {missing_fields}")
        
        # Validate data types
        for i, record in enumerate(json_data[:10]):  # Check first 10 records
            try:
                datetime.strptime(record['date'], '%Y-%m-%d')
                assert isinstance(record['forecast'], (int, float))
                assert isinstance(record['lower_bound'], (int, float))
                assert isinstance(record['upper_bound'], (int, float))
                assert record['granularity'] == granularity
            except (AssertionError, ValueError) as e:
                self.logger.error(f"Invalid data format in record {i}: {str(e)}")
                raise ValueError(f"Invalid data format in record {i}")
        
        self.logger.info(f"JSON output validation passed: {len(json_data)} records")
    
    def generate_summary_report(self, all_forecasts: Dict[str, Dict[str, pd.DataFrame]], 
                              execution_time: float) -> str:
        """Generate a summary report of the forecasting run"""
        report = {
            "run_timestamp": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "configuration": {
                "train_period": f"{self.config.get('train_start')} to {self.config.get('train_end')}",
                "forecast_period": f"{self.config.get('forecast_start')} to {self.config.get('forecast_end')}",
                "granularities": self.config.get('granularities', [])
            },
            "results_summary": {}
        }
        
        for granularity, forecasts in all_forecasts.items():
            report["results_summary"][granularity] = {
                "variables_forecasted": list(forecasts.keys()),
                "forecast_periods": len(list(forecasts.values())[0]) if forecasts else 0
            }
        
        # Save summary report
        filename = f"forecast_summary_{self.run_date}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Summary report saved: {filepath}")
        return filepath