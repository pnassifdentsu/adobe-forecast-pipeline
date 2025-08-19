#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import glob
from typing import Dict, List, Optional

from config import ForecastConfig
from forecast_pipeline import ForecastPipeline

# Page config
st.set_page_config(
    page_title="Adobe SEM Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ForecastDashboard:
    """Streamlit dashboard for Prophet forecast visualization"""
    
    def __init__(self):
        self.config = None
        self.pipeline = None
        
    def load_forecast_data(self, output_dir: str) -> Dict[str, pd.DataFrame]:
        """Load forecast data from JSON files"""
        forecast_data = {}
        
        # Find all JSON forecast files
        json_files = glob.glob(os.path.join(output_dir, "prophet_forecast_*.json"))
        
        for file_path in json_files:
            # Extract granularity from filename
            filename = os.path.basename(file_path)
            if 'daily' in filename:
                granularity = 'daily'
            elif 'weekly' in filename:
                granularity = 'weekly'
            elif 'monthly' in filename:
                granularity = 'monthly'
            else:
                continue
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                forecast_data[granularity] = df
                
        return forecast_data
    
    def load_historical_data(self, config: ForecastConfig) -> pd.DataFrame:
        """Load historical data for comparison"""
        from data_processor import DataProcessor
        
        data_processor = DataProcessor(config)
        raw_data = data_processor.load_data(config.get('data_file'))
        raw_data = data_processor.validate_columns(raw_data)
        processed_data = data_processor.preprocess_data(raw_data)
        
        return processed_data
    
    def aggregate_historical_data(self, historical_data: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Aggregate historical data to match forecast granularity"""
        if granularity == 'daily' or historical_data.empty:
            return historical_data
            
        # Get target columns (excluding date and cost)
        target_columns = []
        for col in historical_data.columns:
            if col not in ['date', 'cost'] and historical_data[col].dtype in ['int64', 'float64']:
                target_columns.append(col)
        
        if not target_columns:
            return historical_data
            
        # Create a copy for aggregation
        agg_data = historical_data.copy()
        
        if granularity == 'weekly':
            # Aggregate to weekly (Monday start)
            agg_data['week_start'] = agg_data['date'].dt.to_period('W-MON').dt.start_time
            
            # Group by week and sum target columns
            weekly_agg = agg_data.groupby('week_start').agg({
                col: 'sum' for col in target_columns
            }).reset_index()
            
            # Rename week_start to date
            weekly_agg = weekly_agg.rename(columns={'week_start': 'date'})
            
            # Add cost as mean if present
            if 'cost' in agg_data.columns:
                cost_agg = agg_data.groupby('week_start')['cost'].mean().reset_index()
                weekly_agg = weekly_agg.merge(cost_agg.rename(columns={'week_start': 'date'}), on='date')
            
            return weekly_agg
            
        elif granularity == 'monthly':
            # Aggregate to monthly
            agg_data['month_start'] = agg_data['date'].dt.to_period('M').dt.start_time
            
            # Group by month and sum target columns
            monthly_agg = agg_data.groupby('month_start').agg({
                col: 'sum' for col in target_columns
            }).reset_index()
            
            # Rename month_start to date
            monthly_agg = monthly_agg.rename(columns={'month_start': 'date'})
            
            # Add cost as mean if present
            if 'cost' in agg_data.columns:
                cost_agg = agg_data.groupby('month_start')['cost'].mean().reset_index()
                monthly_agg = monthly_agg.merge(cost_agg.rename(columns={'month_start': 'date'}), on='date')
            
            return monthly_agg
            
        return historical_data
    
    def create_forecast_chart(self, historical_data: pd.DataFrame, 
                            forecast_data: pd.DataFrame, 
                            variable: str, granularity: str) -> go.Figure:
        """Create interactive forecast chart with historical and forecast data"""
        
        fig = go.Figure()
        
        # Aggregate historical data to match forecast granularity
        aggregated_historical = self.aggregate_historical_data(historical_data, granularity)
        
        # Filter forecast data for the variable to get forecast period dates
        forecast_var = forecast_data[forecast_data['variable'] == variable].copy()
        forecast_var = forecast_var.sort_values('date')
        
        # Determine the cutoff date (first forecast date)
        forecast_start = None
        if not forecast_var.empty:
            forecast_start = forecast_var['date'].min()
        
        # Filter aggregated historical data for the variable and limit to before forecast period
        if variable in aggregated_historical.columns:
            hist_data = aggregated_historical[['date', variable]].copy()
            hist_data = hist_data.sort_values('date')
            
            # Only show historical data up to the forecast start date
            if forecast_start is not None:
                hist_data = hist_data[hist_data['date'] < forecast_start]
            
            # Remove rows where the variable value is 0 (likely missing/filled values)
            hist_data = hist_data[hist_data[variable] > 0]
            
            if not hist_data.empty:
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=hist_data['date'],
                    y=hist_data[variable],
                    name=f'{variable} (Historical)',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: %{y:,.0f}<br>' +
                                 '<extra></extra>'
                ))
        
        if not forecast_var.empty:
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_var['date'], forecast_var['date'][::-1]]),
                y=pd.concat([forecast_var['upper_bound'], forecast_var['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                hoverinfo='skip'
            ))
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast_var['date'],
                y=forecast_var['forecast'],
                name=f'{variable} (Forecast)',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Forecast: %{y:,.0f}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{variable} - {granularity.title()} Forecast',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True,
            height=500,
            plot_bgcolor='white',
            font=dict(size=12)
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor='lightgray',
            gridwidth=1
        )
        fig.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickformat=',.0f'
        )
        
        return fig
    
    def create_summary_metrics(self, forecast_data: pd.DataFrame, variable: str) -> Dict:
        """Calculate summary metrics for a variable"""
        var_data = forecast_data[forecast_data['variable'] == variable]
        
        if var_data.empty:
            return {}
            
        return {
            'avg_forecast': var_data['forecast'].mean(),
            'total_forecast': var_data['forecast'].sum(),
            'min_forecast': var_data['forecast'].min(),
            'max_forecast': var_data['forecast'].max(),
            'confidence_range': (var_data['upper_bound'] - var_data['lower_bound']).mean()
        }
    
    def run_new_forecast(self, config_data: dict) -> bool:
        """Run new forecast with provided configuration"""
        try:
            # Create temporary config
            temp_config = ForecastConfig()
            for key, value in config_data.items():
                temp_config.set(key, value)
            
            # Run pipeline
            pipeline = ForecastPipeline()
            pipeline.config = temp_config
            
            # Initialize components with new config
            from data_processor import DataProcessor
            from prophet_forecaster import ProphetForecaster
            from output_manager import OutputManager
            
            pipeline.data_processor = DataProcessor(temp_config)
            pipeline.forecaster = ProphetForecaster(temp_config)
            pipeline.output_manager = OutputManager(temp_config)
            
            results = pipeline.run_pipeline()
            return results['success']
            
        except Exception as e:
            st.error(f"Forecast failed: {str(e)}")
            return False
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("📈 Forecast Controls")
        
        # Mode selection
        mode = st.sidebar.radio(
            "Mode",
            ["View Existing Results", "Run New Forecast"],
            index=0
        )
        
        if mode == "View Existing Results":
            # Output directory selection
            output_dir = st.sidebar.text_input(
                "Output Directory",
                value="output",
                help="Directory containing forecast JSON files"
            )
            
            return {"mode": "view", "output_dir": output_dir}
            
        else:  # Run New Forecast
            st.sidebar.subheader("Configuration")
            
            # Data file
            data_file = st.sidebar.text_input(
                "Data File",
                value="Data/Adobe Forecast Data 8.17.xlsx",
                help="Path to Excel data file"
            )
            
            # Date ranges
            col1, col2 = st.sidebar.columns(2)
            with col1:
                train_start = st.date_input(
                    "Train Start",
                    value=datetime(2024, 1, 1),
                    help="Training data start date"
                )
                
            with col2:
                train_end = st.date_input(
                    "Train End", 
                    value=datetime.now() - timedelta(days=1),
                    help="Training data end date"
                )
            
            col3, col4 = st.sidebar.columns(2)
            with col3:
                forecast_start = st.date_input(
                    "Forecast Start",
                    value=datetime.now(),
                    help="Forecast start date"
                )
                
            with col4:
                forecast_end = st.date_input(
                    "Forecast End",
                    value=datetime.now() + timedelta(days=30),
                    help="Forecast end date"
                )
            
            # Granularities
            granularities = st.sidebar.multiselect(
                "Granularities",
                ["daily", "weekly", "monthly"],
                default=["daily", "weekly", "monthly"],
                help="Select forecast granularities"
            )
            
            # Target variables
            default_targets = [
                "Visits", "NC Visits", "CC Visits", "Mobile Visits",
                "Orders", "NCOs", "CCOs", "Mobile Orders", 
                "Call Volume", "Call Orders", "NCO Call Orders", "Mobile Call Orders"
            ]
            
            target_columns = st.sidebar.multiselect(
                "Target Variables",
                default_targets,
                default=default_targets,
                help="Select variables to forecast"
            )
            
            # Run forecast button
            run_forecast = st.sidebar.button(
                "🚀 Run Forecast",
                type="primary",
                help="Execute forecast with current settings"
            )
            
            return {
                "mode": "run",
                "data_file": data_file,
                "train_start": train_start.strftime('%Y-%m-%d'),
                "train_end": train_end.strftime('%Y-%m-%d'),
                "forecast_start": forecast_start.strftime('%Y-%m-%d'),
                "forecast_end": forecast_end.strftime('%Y-%m-%d'),
                "granularities": granularities,
                "target_columns": target_columns,
                "run_forecast": run_forecast
            }
    
    def render_main_content(self, sidebar_config: dict):
        """Render main dashboard content"""
        st.title("📈 Adobe SEM Forecast Dashboard")
        st.markdown("---")
        
        if sidebar_config["mode"] == "run":
            if sidebar_config.get("run_forecast"):
                st.info("🔄 Running forecast...")
                
                config_data = {
                    "data_file": sidebar_config["data_file"],
                    "train_start": sidebar_config["train_start"],
                    "train_end": sidebar_config["train_end"],
                    "forecast_start": sidebar_config["forecast_start"],
                    "forecast_end": sidebar_config["forecast_end"],
                    "granularities": sidebar_config["granularities"],
                    "target_columns": sidebar_config["target_columns"],
                    "output_dir": "output"
                }
                
                success = self.run_new_forecast(config_data)
                
                if success:
                    st.success("✅ Forecast completed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Forecast failed. Check logs for details.")
            else:
                st.info("👈 Configure settings in the sidebar and click 'Run Forecast' to generate new predictions.")
                
                # Show configuration preview
                with st.expander("Configuration Preview", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Date Range:**")
                        st.write(f"Training: {sidebar_config['train_start']} to {sidebar_config['train_end']}")
                        st.write(f"Forecast: {sidebar_config['forecast_start']} to {sidebar_config['forecast_end']}")
                        
                    with col2:
                        st.write("**Settings:**")
                        st.write(f"Granularities: {', '.join(sidebar_config['granularities'])}")
                        st.write(f"Variables: {len(sidebar_config['target_columns'])} selected")
            
            return
        
        # View existing results mode
        output_dir = sidebar_config["output_dir"]
        
        if not os.path.exists(output_dir):
            st.warning(f"Output directory '{output_dir}' not found. Please run a forecast first or check the path.")
            return
        
        # Load forecast data
        try:
            forecast_data = self.load_forecast_data(output_dir)
            
            if not forecast_data:
                st.warning("No forecast data found in the output directory. Please run a forecast first.")
                return
                
        except Exception as e:
            st.error(f"Error loading forecast data: {str(e)}")
            return
        
        # Load historical data for comparison
        try:
            config = ForecastConfig("example_config.json")
            historical_data = self.load_historical_data(config)
        except Exception as e:
            st.warning(f"Could not load historical data: {str(e)}")
            historical_data = pd.DataFrame()
        
        # Main dashboard tabs
        tab1, tab2, tab3 = st.tabs(["📊 Interactive Charts", "📋 Summary Metrics", "📈 Raw Data"])
        
        with tab1:
            self.render_charts_tab(forecast_data, historical_data)
            
        with tab2:
            self.render_metrics_tab(forecast_data)
            
        with tab3:
            self.render_data_tab(forecast_data)
    
    def render_charts_tab(self, forecast_data: Dict[str, pd.DataFrame], historical_data: pd.DataFrame):
        """Render interactive charts tab"""
        st.subheader("Interactive Forecast Charts")
        
        # Chart controls
        col1, col2 = st.columns(2)
        
        with col1:
            selected_granularity = st.selectbox(
                "Select Granularity",
                options=list(forecast_data.keys()),
                index=0
            )
        
        with col2:
            # Get available variables for selected granularity
            available_vars = forecast_data[selected_granularity]['variable'].unique()
            selected_variable = st.selectbox(
                "Select Variable",
                options=available_vars,
                index=0
            )
        
        # Create and display chart
        if selected_granularity in forecast_data:
            fig = self.create_forecast_chart(
                historical_data,
                forecast_data[selected_granularity],
                selected_variable,
                selected_granularity
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show chart data details
            with st.expander("Chart Data Details"):
                var_data = forecast_data[selected_granularity][
                    forecast_data[selected_granularity]['variable'] == selected_variable
                ]
                st.dataframe(var_data, use_container_width=True)
    
    def render_metrics_tab(self, forecast_data: Dict[str, pd.DataFrame]):
        """Render summary metrics tab"""
        st.subheader("Forecast Summary Metrics")
        
        for granularity, data in forecast_data.items():
            st.write(f"### {granularity.title()} Forecasts")
            
            variables = data['variable'].unique()
            
            # Create metrics grid
            cols = st.columns(min(len(variables), 4))
            
            for i, variable in enumerate(variables):
                with cols[i % 4]:
                    metrics = self.create_summary_metrics(data, variable)
                    
                    if metrics:
                        st.metric(
                            label=variable,
                            value=f"{metrics['avg_forecast']:,.0f}",
                            delta=f"±{metrics['confidence_range']:,.0f}"
                        )
                        
                        with st.expander(f"{variable} Details"):
                            st.write(f"**Total Forecast:** {metrics['total_forecast']:,.0f}")
                            st.write(f"**Min:** {metrics['min_forecast']:,.0f}")
                            st.write(f"**Max:** {metrics['max_forecast']:,.0f}")
            
            st.markdown("---")
    
    def render_data_tab(self, forecast_data: Dict[str, pd.DataFrame]):
        """Render raw data tab"""
        st.subheader("Raw Forecast Data")
        
        for granularity, data in forecast_data.items():
            with st.expander(f"{granularity.title()} Data ({len(data)} records)", expanded=False):
                st.dataframe(data, use_container_width=True)
                
                # Download button
                csv = data.to_csv(index=False)
                st.download_button(
                    label=f"Download {granularity} CSV",
                    data=csv,
                    file_name=f"forecast_{granularity}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

def main():
    """Main dashboard application"""
    dashboard = ForecastDashboard()
    
    # Render sidebar and get configuration
    sidebar_config = dashboard.render_sidebar()
    
    # Render main content
    dashboard.render_main_content(sidebar_config)

if __name__ == "__main__":
    main()