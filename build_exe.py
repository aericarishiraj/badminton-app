#!/usr/bin/env python3
"""
Build script for creating executable from badminton app
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_requirements():
    """Install required packages including PyInstaller"""
    print("Installing requirements...")
    
    try:
        import PyInstaller
        print("PyInstaller is already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    if os.path.exists("requirements.txt"):
        print("Installing requirements from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_spec_file():
    """Create PyInstaller spec file"""
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
        'pyreadline3',
        'colorama',
        'wcwidth',
        'pycparser',
        'cffi',
        'pywin32',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BadmintonApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''
    
    with open('badminton_app.spec', 'w') as f:
        f.write(spec_content)
    
    print("Created badminton_app.spec file")

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building executable...")
    
    # Clean previous builds
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    
    # Build using spec file
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "badminton_app.spec"])
    
    print("Build completed successfully!")
    print("Executable location: dist/BadmintonApp.exe")

def create_launcher_script():
    """Create a launcher script that installs requirements and runs the app"""
    launcher_content = '''@echo off
echo Installing requirements...
pip install -r requirements.txt
echo Starting Badminton App...
start "" "dist\\BadmintonApp.exe"
'''
    
    with open('run_badminton_app.bat', 'w') as f:
        f.write(launcher_content)
    
    print("Created run_badminton_app.bat launcher script")

def main():
    """Main build process"""
    print("Starting build process for Badminton App...")
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Create spec file
    create_spec_file()
    
    # Step 3: Build executable
    build_executable()
    
    # Step 4: Create launcher script
    create_launcher_script()
    
    print("\n" + "="*50)
    print("BUILD COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Files created:")
    print("- dist/BadmintonApp.exe (main executable)")
    print("- run_badminton_app.bat (launcher script)")
    print("\nTo run the app:")
    print("1. Double-click run_badminton_app.bat")
    print("2. Or directly run dist/BadmintonApp.exe")
    print("\nThe app will automatically install requirements and start the web server.")

if __name__ == "__main__":
    main() 