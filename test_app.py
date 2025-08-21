#!/usr/bin/env python3
"""
Simple test script to verify the badminton app works before building
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn
        print("✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Seaborn import failed: {e}")
        return False
    
    try:
        import scipy
        print("✓ SciPy imported successfully")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    try:
        import networkx
        print("✓ NetworkX imported successfully")
    except ImportError as e:
        print(f"✗ NetworkX import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    return True

def test_app_import():
    """Test if the main app can be imported"""
    print("\nTesting app import...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from app import app
        print("✓ Main app imported successfully")
        
        from controllers.dashboard_controller import get_all_players
        print("✓ Controllers imported successfully")
        
        from models.data_model import load_data
        print("✓ Models imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ App import failed: {e}")
        return False

def test_data_loading():
    """Test if data can be loaded"""
    print("\nTesting data loading...")
    
    try:
        from models.data_model import load_data
        df = load_data()
        print(f"✓ Data loaded successfully. Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("BADMINTON APP PRE-BUILD TEST")
    print("="*50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_app_import():
        all_passed = False
    
    if not test_data_loading():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ ALL TESTS PASSED! Ready to build.")
        print("Run: python3 build_mac.py")
    else:
        print("✗ SOME TESTS FAILED! Fix issues before building.")
    print("="*50)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
