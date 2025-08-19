#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    
    # Check if we're in the right directory
    if not os.path.exists('dashboard.py'):
        print("Error: dashboard.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Launch Streamlit
    try:
        print("Launching Adobe SEM Forecast Dashboard...")
        print("Dashboard will open in your default browser")
        print("URL: http://localhost:8501")
        print("\nPress Ctrl+C to stop the dashboard")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port=8501",
            "--server.headless=false",
            "--browser.gatherUsageStats=false"
        ])
        
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()