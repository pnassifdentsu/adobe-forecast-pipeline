#!/usr/bin/env python3
"""
Validation script to check if the Prophet forecasting pipeline is properly set up.
Run this script after installing requirements to verify the installation.
"""

import sys
import os

def check_imports():
    """Check if all required modules can be imported"""
    print("🔍 Checking imports...")
    
    try:
        from config import ForecastConfig
        print("✅ config.py imports successfully")
        
        from data_processor import DataProcessor  
        print("✅ data_processor.py imports successfully")
        
        from prophet_forecaster import ProphetForecaster
        print("✅ prophet_forecaster.py imports successfully")
        
        from output_manager import OutputManager
        print("✅ output_manager.py imports successfully")
        
        from forecast_pipeline import ForecastPipeline
        print("✅ forecast_pipeline.py imports successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\\n🔍 Checking dependencies...")
    
    dependencies = [
        'pandas', 'numpy', 'matplotlib', 'prophet', 'openpyxl', 'seaborn'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is installed")
        except ImportError:
            print(f"❌ {dep} is NOT installed")
            missing.append(dep)
    
    return len(missing) == 0, missing

def check_configuration():
    """Check configuration functionality"""
    print("\\n🔍 Checking configuration...")
    
    try:
        from config import ForecastConfig
        config = ForecastConfig()
        
        target_columns = config.get('target_columns', [])
        print(f"✅ Configuration loaded with {len(target_columns)} target columns")
        
        # Test configuration methods
        config.set('test_key', 'test_value')
        test_value = config.get('test_key')
        
        if test_value == 'test_value':
            print("✅ Configuration set/get methods work correctly")
            return True
        else:
            print("❌ Configuration set/get methods failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False

def check_files():
    """Check if all required files exist"""
    print("\\n🔍 Checking required files...")
    
    required_files = [
        'config.py',
        'data_processor.py', 
        'prophet_forecaster.py',
        'output_manager.py',
        'forecast_pipeline.py',
        'requirements.txt',
        'example_config.json',
        'README.md',
        'CLAUDE.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} is missing")
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def main():
    """Main validation function"""
    print("🚀 Prophet Forecasting Pipeline - Setup Validation")
    print("=" * 60)
    
    # Check files first
    files_ok, missing_files = check_files()
    
    if not files_ok:
        print(f"\\n❌ Setup validation failed - missing files: {missing_files}")
        return False
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    
    if not deps_ok:
        print(f"\\n⚠️  Missing dependencies: {missing_deps}")
        print("Run: pip install -r requirements.txt")
        print("Then run this validation script again.")
        return False
    
    # Check imports
    imports_ok = check_imports()
    
    if not imports_ok:
        return False
    
    # Check configuration
    config_ok = check_configuration()
    
    if not config_ok:
        return False
    
    print("\\n" + "=" * 60)
    print("🎉 Setup validation passed! Pipeline is ready to use.")
    print("\\nNext steps:")
    print("1. Place your Excel data file in the Data/ directory")
    print("2. Run: python forecast_pipeline.py")
    print("3. Check the output/ directory for results")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)