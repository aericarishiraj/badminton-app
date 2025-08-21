# Badminton Analysis App

A Flask-based web application for analyzing badminton match data with interactive visualizations and dashboards.

## Features

- Match analysis dashboards
- Shot landing heatmaps
- Player performance comparisons
- Statistical radar charts
- Rally analysis
- Error zone mapping
- Player rankings and networks

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   python app.py
   ```

3. Open http://localhost:5000 in your browser

## Deployment on Render

This app is configured for easy deployment on Render.com

### Automatic Deployment

1. Push your code to GitHub
2. Sign up at [render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will automatically detect the configuration from `render.yaml`
6. Click "Create Web Service"

### Manual Configuration

If you prefer manual setup:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Environment**: Python 3.11

## Data

The app uses CSV data files stored in the `static/` directory. Make sure these files are included in your repository for the app to function properly.

## Requirements

- Python 3.11+
- Flask 3.0.0
- NumPy, Pandas, Matplotlib
- OpenCV, NetworkX, SciPy
- Gunicorn (for production)
# badminton-app
# badminton-app
