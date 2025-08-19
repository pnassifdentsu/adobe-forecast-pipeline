# Adobe SEM Forecast Pipeline

A configurable Prophet-based forecasting pipeline for SEM time series data, supporting repeatable and extensible forecasting runs.

## Features

- **Prophet-based forecasting** with external regressors (date, cost)
- **Multiple granularities**: Daily, weekly (Monday-Sunday), and monthly forecasts
- **Configurable date ranges** for training and forecasting periods
- **JSON output** with standardized schema
- **PNG visualizations** with confidence intervals
- **Comprehensive validation** and logging throughout the pipeline
- **Missing value imputation** with warning logs
- **Seasonality and trend changepoint support**

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with default configuration:**
   ```bash
   python forecast_pipeline.py
   ```

3. **Run with custom date ranges:**
   ```bash
   python forecast_pipeline.py --train-start 2023-01-01 --train-end 2025-08-16 --forecast-start 2025-08-17 --forecast-end 2025-09-30
   ```

4. **Run with custom configuration:**
   ```bash
   python forecast_pipeline.py --config example_config.json
   ```

## Data Requirements

Input data should contain the following columns with daily granularity:
- `date`: Date column
- `cost`: Cost data (used as external regressor)
- Target variables: `Visits`, `NC Visits`, `CC Visits`, `Mobile Visits`, `Orders`, `NCOs`, `CCOs`, `Mobile Orders`, `Call Volume`, `Call Orders`, `NCO Call Orders`, `Mobile Call Orders`

## Output Format

### JSON Schema
Each JSON output file contains forecasts with the following structure:
```json
{
  "date": "YYYY-MM-DD",
  "granularity": "daily|weekly|monthly",
  "variable": "variable_name",
  "forecast": 123.45,
  "lower_bound": 100.00,
  "upper_bound": 150.00
}
```

### File Naming Convention
- JSON: `prophet_forecast_<granularity>_<run_date>.json`
- PNG: `prophet_forecast_<variable>_<granularity>_<run_date>.png`

## Configuration

Create a JSON configuration file to customize the pipeline:

```json
{
  "data_file": "Data/Adobe Forecast Data 8.17.xlsx",
  "train_start": "2023-01-01",
  "train_end": "2025-08-16",
  "forecast_start": "2025-08-17",
  "forecast_end": "2025-09-30",
  "output_dir": "output",
  "granularities": ["daily", "weekly", "monthly"],
  "prophet_params": {
    "yearly_seasonality": true,
    "weekly_seasonality": true,
    "changepoint_prior_scale": 0.05,
    "interval_width": 0.95
  }
}
```

## Testing

Run the test suite to verify the pipeline:
```bash
python test_pipeline.py
```

## Architecture

- `config.py`: Configuration management
- `data_processor.py`: Data preprocessing and validation
- `prophet_forecaster.py`: Prophet model implementation
- `output_manager.py`: JSON and visualization generation
- `forecast_pipeline.py`: Main orchestration script
- `test_pipeline.py`: Comprehensive test suite