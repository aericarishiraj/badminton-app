#!/bin/bash
echo "Starting Badminton Analysis App..."
echo "Opening app bundle..."

# Check if the app bundle exists
if [ -d "dist/BadmintonAnalysisApp.app" ]; then
    echo "App bundle found. Opening..."
    open "dist/BadmintonAnalysisApp.app"
else
    echo "App bundle not found. Please run the build script first."
    echo "Run: python3 build_mac.py"
fi
