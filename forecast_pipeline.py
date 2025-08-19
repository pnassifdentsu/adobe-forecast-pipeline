#!/usr/bin/env python3

import logging
import os
import sys
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from config import ForecastConfig
from data_processor import DataProcessor
from prophet_forecaster import ProphetForecaster
from output_manager import OutputManager


class ForecastPipeline:
    """Main orchestration class for the Prophet forecasting pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config = ForecastConfig(config_path)
        self.setup_logging()
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.forecaster = ProphetForecaster(self.config)
        self.output_manager = OutputManager(self.config)
        
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'forecast_pipeline_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete forecasting pipeline"""
        start_time = time.time()
        results = {
            'success': False,
            'forecasts': {},
            'output_files': {},
            'execution_time': 0,
            'errors': []
        }
        
        try:
            self.logger.info("Starting Prophet forecasting pipeline")
            
            # Step 1: Load and validate data
            self.logger.info("Step 1: Loading and preprocessing data")
            raw_data = self.data_processor.load_data(self.config.get('data_file'))
            raw_data = self.data_processor.validate_columns(raw_data)
            processed_data = self.data_processor.preprocess_data(raw_data)
            self.data_processor.validate_results(processed_data, "Data preprocessing")
            
            # Step 2: Filter training data
            self.logger.info("Step 2: Filtering training data")
            train_data = self.data_processor.filter_date_range(
                processed_data,
                self.config.get('train_start'),
                self.config.get('train_end')
            )
            self.data_processor.validate_results(train_data, "Training data filtering")
            
            # Step 3: Generate forecasts for each granularity
            all_forecasts = {}
            all_output_files = {}
            daily_forecasts = None  # Store daily forecasts for monthly combination
            
            granularities = self.config.get('granularities', ['daily'])
            
            for granularity in granularities:
                self.logger.info(f"Step 3.{granularities.index(granularity) + 1}: Processing {granularity} forecasts")
                
                if granularity == "monthly":
                    # Special handling for monthly forecasts - use daily forecasts to create complete months
                    if daily_forecasts is None:
                        # Generate daily forecasts first if not already done
                        daily_train_data = self.data_processor.aggregate_to_granularity(train_data, 'daily')
                        daily_forecasts = self.forecaster.forecast_all_variables(daily_train_data, 'daily')
                    
                    # Create complete monthly forecasts combining partial historical + daily forecasts
                    forecasts = self.data_processor.create_complete_monthly_forecast(
                        train_data, daily_forecasts, self.config.get('train_end')
                    )
                    
                    # Use monthly aggregated historical data for visualization
                    granular_train_data = self.data_processor.aggregate_to_granularity(train_data, granularity)
                    self.data_processor.validate_results(granular_train_data, f"{granularity} aggregation")
                    
                else:
                    # Normal processing for daily and weekly
                    granular_train_data = self.data_processor.aggregate_to_granularity(train_data, granularity)
                    self.data_processor.validate_results(granular_train_data, f"{granularity} aggregation")
                    
                    # Generate forecasts for all variables
                    forecasts = self.forecaster.forecast_all_variables(granular_train_data, granularity)
                    
                    # Store daily forecasts for potential monthly use
                    if granularity == "daily":
                        daily_forecasts = forecasts
                
                # Validate each forecast
                for variable, forecast_df in forecasts.items():
                    if not forecast_df.empty:
                        self.forecaster.validate_forecast(forecast_df, variable, granularity)
                
                all_forecasts[granularity] = forecasts
                
                # Step 4: Generate outputs
                self.logger.info(f"Step 4.{granularities.index(granularity) + 1}: Generating {granularity} outputs")
                
                # Format and save JSON output
                json_data = self.output_manager.format_forecast_to_json(forecasts, granularity)
                self.output_manager.validate_json_output(json_data, granularity)
                json_file = self.output_manager.save_json_output(json_data, granularity)
                
                # Generate visualizations
                if granularity == "monthly":
                    # For monthly, create enhanced historical data that includes partial current month
                    enhanced_monthly_historical = self.create_enhanced_monthly_historical(
                        train_data, self.config.get('train_end')
                    )
                    viz_files = self.output_manager.generate_all_visualizations(
                        forecasts, enhanced_monthly_historical, granularity
                    )
                else:
                    viz_files = self.output_manager.generate_all_visualizations(
                        forecasts, granular_train_data, granularity
                    )                
                all_output_files[granularity] = {
                    'json': json_file,
                    'visualizations': viz_files
                }
            
            # Step 5: Generate summary report
            execution_time = time.time() - start_time
            summary_file = self.output_manager.generate_summary_report(all_forecasts, execution_time)
            
            results.update({
                'success': True,
                'forecasts': all_forecasts,
                'output_files': all_output_files,
                'summary_file': summary_file,
                'execution_time': execution_time
            })
            
            self.logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            results.update({
                'success': False,
                'execution_time': execution_time,
                'errors': [error_msg]
            })
            raise
        
        return results
    
    def create_enhanced_monthly_historical(self, daily_data: pd.DataFrame, train_end_date: str) -> pd.DataFrame:
        """Create enhanced monthly historical data for visualization that includes complete months"""
        train_end = pd.to_datetime(train_end_date)
        
        # Filter daily data from 2024-01-01 (for visualization purposes)
        viz_start = pd.to_datetime('2024-01-01')
        filtered_daily = daily_data[daily_data['date'] >= viz_start].copy()
        
        # If training ends mid-month, we want to show complete months up to the previous month
        # and then show the partial current month separately or combine it with forecast
        if train_end.day < 28:  # Mid-month
            # Show complete months up to previous month end
            prev_month_end = (train_end.replace(day=1) - pd.Timedelta(days=1))
            complete_months_data = filtered_daily[filtered_daily['date'] <= prev_month_end].copy()
            
            # Aggregate complete months
            if not complete_months_data.empty:
                complete_months_data['month_start'] = complete_months_data['date'].dt.to_period('M').dt.start_time
                monthly_complete = complete_months_data.groupby('month_start').agg({
                    col: 'sum' for col in self.config.get('target_columns', []) if col in complete_months_data.columns
                }).reset_index()
                monthly_complete = monthly_complete.rename(columns={'month_start': 'date'})
                
                # Add cost as mean
                if 'cost' in complete_months_data.columns:
                    cost_agg = complete_months_data.groupby('month_start')['cost'].mean().reset_index()
                    monthly_complete = monthly_complete.merge(cost_agg.rename(columns={'month_start': 'date'}), on='date')
                
                self.logger.info(f"Created enhanced monthly historical data: {len(monthly_complete)} complete months for visualization")
                return monthly_complete
        
        # If training ends at month end or we don't have enough data, use regular monthly aggregation
        filtered_daily['month_start'] = filtered_daily['date'].dt.to_period('M').dt.start_time
        monthly_all = filtered_daily.groupby('month_start').agg({
            col: 'sum' for col in self.config.get('target_columns', []) if col in filtered_daily.columns
        }).reset_index()
        monthly_all = monthly_all.rename(columns={'month_start': 'date'})
        
        # Add cost as mean
        if 'cost' in filtered_daily.columns:
            cost_agg = filtered_daily.groupby('month_start')['cost'].mean().reset_index()
            monthly_all = monthly_all.merge(cost_agg.rename(columns={'month_start': 'date'}), on='date')
        
        return monthly_all
    
    def run_with_custom_dates(self, train_start: str, train_end: str, 
                             forecast_start: str, forecast_end: str) -> Dict[str, Any]:
        """Run pipeline with custom date ranges"""
        self.config.set('train_start', train_start)
        self.config.set('train_end', train_end)
        self.config.set('forecast_start', forecast_start)
        self.config.set('forecast_end', forecast_end)
        
        self.logger.info(f"Running pipeline with custom dates:")
        self.logger.info(f"  Training: {train_start} to {train_end}")
        self.logger.info(f"  Forecast: {forecast_start} to {forecast_end}")
        
        return self.run_pipeline()


def main():
    """Main entry point for command line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prophet-based SEM Forecasting Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--train-start', type=str, help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--forecast-start', type=str, help='Forecast start date (YYYY-MM-DD)')
    parser.add_argument('--forecast-end', type=str, help='Forecast end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = ForecastPipeline(args.config)
        
        # Run with custom dates if provided
        if all([args.train_start, args.train_end, args.forecast_start, args.forecast_end]):
            results = pipeline.run_with_custom_dates(
                args.train_start, args.train_end,
                args.forecast_start, args.forecast_end
            )
        else:
            results = pipeline.run_pipeline()
        
        # Print summary
        if results['success']:
            print(f"\\n[SUCCESS] Pipeline completed successfully!")
            print(f"[TIME] Execution time: {results['execution_time']:.2f} seconds")
            print(f"[DATA] Granularities processed: {list(results['forecasts'].keys())}")
            print(f"[OUTPUT] Output directory: {pipeline.output_manager.output_dir}")
            
            # Print output files
            for granularity, files in results['output_files'].items():
                print(f"\\n{granularity.title()} outputs:")
                print(f"  [JSON] {os.path.basename(files['json'])}")
                viz_files = files['visualizations']
                if len(viz_files) == 1 and viz_files[0].endswith('.pdf'):
                    print(f"  [PDF] {os.path.basename(viz_files[0])}")
                else:
                    print(f"  [VIZ] {len(viz_files)} visualization files")
        else:
            print(f"\\n[X] Pipeline failed!")
            for error in results.get('errors', []):
                print(f"  Error: {error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\\n[X] Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()