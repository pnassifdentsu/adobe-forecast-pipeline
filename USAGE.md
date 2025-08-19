# How to Use the Prophet Forecasting Pipeline

## Quick Start (Recommended for first run)

**Test with just 2 variables and daily granularity:**
```bash
py forecast_pipeline.py --config quick_test_config.json
```

This will:
- Train on data from 2025-01-01 to 2025-08-15
- Forecast from 2025-08-16 to 2025-08-25 (10 days)
- Only process 2 variables: Visits and Orders
- Only generate daily forecasts
- Complete in ~2-3 minutes

## Full Pipeline

**Run with all variables and granularities:**
```bash
py forecast_pipeline.py
```

This processes all 12 variables × 3 granularities = 36 models (~15-20 minutes)

## Custom Date Ranges

```bash
py forecast_pipeline.py --train-start 2024-01-01 --train-end 2024-12-31 --forecast-start 2025-01-01 --forecast-end 2025-01-31
```

## Output Files

Results are saved to the `output/` directory:
- **JSON files**: `prophet_forecast_{granularity}_{timestamp}.json`
- **PDF charts**: `prophet_forecast_all_variables_{granularity}_{timestamp}.pdf` (one PDF per granularity with all variables)
- **Summary**: `forecast_summary_{timestamp}.json`

### PDF Charts
- Each PDF contains all forecast variables for that granularity in a multi-page layout
- Historical data shown from 2024-01-01 onwards
- Forecast period clearly marked with confidence intervals
- Up to 12 variables per page in a 3x4 grid

### Monthly Forecasts - Mid-Month Intelligence
**Special Feature**: When training data ends mid-month, the pipeline automatically:
- Combines partial historical data from the current month with forecasted data
- Shows expected **end-of-month (EOM) performance** 
- Example: Training ends Aug 15 → combines Aug 1-15 actual + Aug 16-31 forecast = complete August forecast

This gives you complete monthly insights even when running forecasts mid-month!

## JSON Output Format

Each JSON file contains records like:
```json
{
  "date": "2025-08-16",
  "granularity": "daily",
  "variable": "Visits",
  "forecast": 12345.67,
  "lower_bound": 10000.00,
  "upper_bound": 15000.00
}
```

## Tips

1. **First Run**: Use `quick_test_config.json` to verify everything works
2. **Faster Testing**: Reduce target_columns in config to 2-3 variables
3. **File Issues**: If you get permission errors, close Excel files
4. **Performance**: Full pipeline takes 15-20 minutes for all variables

## Configuration Options

Edit `quick_test_config.json` or create your own:
- `target_columns`: Which variables to forecast
- `granularities`: ["daily", "weekly", "monthly"]
- `train_start/end`: Training data period
- `forecast_start/end`: Forecast period
- `prophet_params`: Model settings (seasonality, etc.)

## Troubleshooting

- **Permission Error**: Close Excel files, try local copy
- **Empty Output**: Check training data has enough history (>30 days recommended)
- **Slow Performance**: Reduce target_columns or granularities
- **Encoding Error**: The pipeline handles this automatically now