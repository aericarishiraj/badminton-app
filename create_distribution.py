#!/usr/bin/env python3
"""
Create a complete distribution package for the Badminton App
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_distribution():
    """Create a complete distribution package"""
    
    dist_dir = "BadmintonApp_Distribution"
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    os.makedirs(dist_dir)
    
    print("Copying executable...")
    shutil.copy2("dist/BadmintonApp.exe", dist_dir)
    
    print("Copying static files...")
    shutil.copytree("static", os.path.join(dist_dir, "static"))
    
    print("Copying templates...")
    shutil.copytree("templates", os.path.join(dist_dir, "templates"))
    
    print("Copying models...")
    shutil.copytree("models", os.path.join(dist_dir, "models"))
    
    print("Copying controllers...")
    shutil.copytree("controllers", os.path.join(dist_dir, "controllers"))
    
    print("Copying requirements...")
    shutil.copy2("requirements.txt", dist_dir)
    
    print("Copying launcher script...")
    shutil.copy2("run_badminton_app.bat", dist_dir)
    
    create_distribution_readme(dist_dir)
    
    print("Creating ZIP archive...")
    create_zip_archive(dist_dir)
    
    print(f"\nDistribution created successfully!")
    print(f"Location: {dist_dir}")
    print(f"ZIP file: {dist_dir}.zip")
    print("\nTo distribute:")
    print(f"1. Copy the '{dist_dir}' folder to any computer")
    print(f"2. Or share the '{dist_dir}.zip' file")
    print("3. Run 'run_badminton_app.bat' on the target computer")

def create_distribution_readme(dist_dir):
    """Create a README file for the distribution"""
    readme_content = """# Badminton Analysis App - Distribution Package

## Quick Start

1. **Install Python** (if not already installed)
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. **Run the Application**
   - Double-click `run_badminton_app.bat`
   - OR double-click `BadmintonApp.exe` directly

3. **Access the Web Interface**
   - Open your web browser
   - Go to: http://localhost:5000

## What's Included

- `BadmintonApp.exe` - Main application executable
- `run_badminton_app.bat` - Easy launcher script
- `requirements.txt` - Python dependencies
- `static/` - Data files and assets
- `templates/` - Web interface templates
- `models/` - Data processing modules
- `controllers/` - Application logic

## System Requirements

- Windows 10 or later
- Python 3.7 or higher
- Internet connection (for first-time setup)

## Features

- **Automatic Setup**: Installs required packages automatically
- **Web Interface**: Access via any web browser
- **Data Analysis**: Comprehensive badminton match analysis
- **Visualizations**: Interactive charts and heatmaps
- **Player Rankings**: Statistical player comparisons

## Troubleshooting

### "Python not found" error
- Install Python from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation

### "Module not found" error
- The app will automatically install missing packages
- If manual installation is needed, run: `pip install -r requirements.txt`

### Web interface doesn't load
- Check that the app is running (console window should be open)
- Try accessing http://127.0.0.1:5000 instead
- Check Windows firewall settings

### Port already in use
- Close other applications using port 5000
- Or restart your computer

## Support

For issues or questions:
1. Check the console output for error messages
2. Ensure Python is properly installed
3. Verify internet connection for package installation
4. Check that all files are present in the distribution folder

## Security

- This is a local application only
- No data is sent to external servers
- All processing happens on your computer
- Web interface is only accessible locally
"""
    
    with open(os.path.join(dist_dir, "README.txt"), 'w') as f:
        f.write(readme_content)

def create_zip_archive(dist_dir):
    """Create a ZIP file of the distribution"""
    zip_filename = f"{dist_dir}.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dist_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dist_dir)
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    create_distribution() 