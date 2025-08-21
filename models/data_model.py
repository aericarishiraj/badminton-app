import pandas as pd
import os
import sys

# Handle PyInstaller bundled vs development paths
if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

if getattr(sys, 'frozen', False):
    CSV_PATH = os.path.join(application_path, 'static', 'combined_match_data_translated.csv')
else:
    CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'combined_match_data_translated.csv'))

def load_data():
    return pd.read_csv(CSV_PATH)

def load_match_summary():
    df = load_data()
    summary = df[['tournament', 'year', 'round', 'winner', 'loser']].drop_duplicates()
    summary['match'] = summary['winner'] + ' vs ' + summary['loser']
    return summary.to_dict(orient='records')