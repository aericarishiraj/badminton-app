#!/usr/bin/env python3
"""
Build script for creating macOS distributable from badminton app
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✓ PyInstaller is already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def create_mac_spec_file():
    """Create PyInstaller spec file optimized for macOS"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('models', 'models'),
        ('controllers', 'controllers'),
        ('requirements.txt', '.'),
    ],
    hiddenimports=[
        'flask',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'networkx',
        'cv2',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.figure',
        'matplotlib.pyplot',
        'scipy.stats',
        'scipy.spatial',
        'scipy.ndimage',
        'networkx.algorithms',
        'networkx.drawing',
        'pandas._libs',
        'pandas.core',
        'pandas.io',
        'jinja2',
        'werkzeug',
        'markupsafe',
        'itsdangerous',
        'click',
        'blinker',
        'six',
        'python_dateutil',
        'pytz',
        'pyparsing',
        'packaging',
        'kiwisolver',
        'fonttools',
        'cycler',
        'contourpy',
        'pillow',
        'pycparser',
        'cffi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BadmintonAnalysisApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BadmintonAnalysisApp',
)

app = BUNDLE(
    coll,
    name='BadmintonAnalysisApp.app',
    info_plist={
        'CFBundleName': 'Badminton Analysis App',
        'CFBundleDisplayName': 'Badminton Analysis App',
        'CFBundleIdentifier': 'com.badminton.analysis.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13.0',
        'CFBundleExecutable': 'BadmintonAnalysisApp',
        'CFBundleIconFile': '',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '????',
        'NSAppleEventsUsageDescription': 'This app needs to access system events to function properly.',
    },
)
'''
    
    with open('badminton_app_mac.spec', 'w') as f:
        f.write(spec_content)
    
    print("✓ Created badminton_app_mac.spec file for macOS")

def build_macos_app():
    """Build the macOS app bundle using PyInstaller"""
    print("Building macOS app bundle...")
    
    # Clean previous builds
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    
    # Build using Mac-specific spec file
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "badminton_app_mac.spec"])
    
    print("✓ Build completed successfully!")
    print("App bundle location: dist/BadmintonAnalysisApp.app")

def create_simple_launcher():
    """Create a simple launcher script that opens the app"""
    launcher_content = '''#!/bin/bash
echo "🚀 Starting Badminton Analysis App..."

# Check if the app bundle exists
if [ -d "dist/BadmintonAnalysisApp.app" ]; then
    echo "✓ App found! Opening..."
    open "dist/BadmintonAnalysisApp.app"
else
    echo "❌ App not found. Building it now..."
    python3 build_mac.py
    if [ -d "dist/BadmintonAnalysisApp.app" ]; then
        echo "✓ Build successful! Opening app..."
        open "dist/BadmintonAnalysisApp.app"
    else
        echo "❌ Build failed. Please check the error messages above."
    fi
fi
'''
    
    with open('run_app.sh', 'w') as f:
        f.write(launcher_content)
    
    # Make the script executable
    os.chmod('run_app.sh', 0o755)
    
    print("✓ Created simple launcher: run_app.sh")

def main():
    """Main build process for macOS"""
    print("🚀 Building Badminton App for macOS...")
    
    # Check if we're on macOS
    if sys.platform != "darwin":
        print("⚠️  Warning: This script is designed for macOS. You're running on:", sys.platform)
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Build cancelled.")
            return
    
    # Step 1: Install PyInstaller only
    install_pyinstaller()
    
    # Step 2: Create Mac-specific spec file
    create_mac_spec_file()
    
    # Step 3: Build macOS app bundle
    build_macos_app()
    
    # Step 4: Create simple launcher
    create_simple_launcher()
    
    print("\n" + "🎉 SUCCESS! Your macOS app is ready!")
    print("="*50)
    print("📱 To run the app:")
    print("   ./run_app.sh")
    print("   OR double-click BadmintonAnalysisApp.app in Finder")
    print("\n💡 The app will automatically install any missing dependencies when it runs.")

if __name__ == "__main__":
    main()
