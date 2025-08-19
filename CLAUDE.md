# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Prophet-based SEM forecasting pipeline for Adobe data that generates configurable, repeatable forecasts with multiple granularities and comprehensive output formats.

## Commands

### Running the Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Run with default configuration
python forecast_pipeline.py

# Run with custom date ranges
python forecast_pipeline.py --train-start 2023-01-01 --train-end 2025-08-16 --forecast-start 2025-08-17 --forecast-end 2025-09-30

# Run with custom configuration file
python forecast_pipeline.py --config example_config.json

# Run tests
python test_pipeline.py
```

### Running the Dashboard
```bash
# Launch interactive Streamlit dashboard
python run_dashboard.py

# Or run directly with Streamlit
streamlit run dashboard.py
```

## Architecture

The pipeline follows a modular architecture:

- **config.py**: Configuration management with JSON support and defaults
- **data_processor.py**: Data loading, preprocessing, validation, and granularity aggregation  
- **prophet_forecaster.py**: Prophet models with external regressors (cost, date features)
- **output_manager.py**: JSON output formatting and PDF visualization generation (single PDF per granularity)
- **forecast_pipeline.py**: Main orchestration with logging and error handling
- **dashboard.py**: Interactive Streamlit dashboard for forecast visualization and analysis
- **run_dashboard.py**: Dashboard launcher script
- **test_pipeline.py**: Comprehensive test suite with sample data generation

## Data Flow

1. **Data Loading**: Excel → pandas DataFrame with validation
2. **Preprocessing**: Date conversion, missing value imputation, filtering
3. **Granularity Aggregation**: Daily → Weekly (Monday start) → Monthly
4. **Prophet Modeling**: Fit models with external regressors for each target variable
5. **Forecasting**: Generate predictions with confidence intervals
6. **Output Generation**: JSON files (standardized schema) + PDF visualizations (all variables per granularity)

## Key Features

- **External Regressors**: Uses 'date' and 'cost' columns as Prophet regressors
- **Missing Value Handling**: Forward-fill imputation with warning logs
- **Multiple Granularities**: Daily, weekly (Monday-Sunday), monthly outputs
- **Validation Pipeline**: Step-by-step validation with corrective action logging
- **Configurable Seasonality**: Supports yearly/weekly seasonality and trend changepoints
- **Standardized Output**: JSON schema compliance with required field validation
- **Mid-Month Intelligence**: Monthly forecasts combine partial historical + forecast data for complete EOM performance
- **Interactive Dashboard**: Streamlit-based web interface for forecast visualization, analysis, and configuration

## Target Variables

Forecasts are generated for all SEM metrics: Visits, NC Visits, CC Visits, Mobile Visits, Orders, NCOs, CCOs, Mobile Orders, Call Volume, Call Orders, NCO Call Orders, Mobile Call Orders