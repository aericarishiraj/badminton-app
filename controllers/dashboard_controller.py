from flask import render_template, request, jsonify
from models.data_model import load_data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from scipy.stats import poisson, skellam
import math
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import random

# Court dimensions constants
COURT_WIDTH = 6.1
COURT_LENGTH = 13.4
SINGLES_WIDTH = 5.18
SINGLES_MARGIN = (COURT_WIDTH - SINGLES_WIDTH) / 2
NET_POSITION = COURT_LENGTH / 2
MIN_DISTANCE_FROM_NET = 0.5

def validate_shot_coordinates(hit_x, hit_y, landing_x, landing_y, player_code, court_rotation_90=False):
    """Validate badminton shot coordinates"""
    warnings = []
    
    # Handle coordinate system rotation (90-degree rotation swaps X/Y axes)
    if court_rotation_90:
        court_x_max = COURT_LENGTH
        court_y_max = COURT_WIDTH
        singles_y_min = SINGLES_MARGIN
        singles_y_max = COURT_WIDTH - SINGLES_MARGIN
        net_x = NET_POSITION
    else:
        court_x_max = COURT_WIDTH
        court_y_max = COURT_LENGTH
        singles_x_min = SINGLES_MARGIN
        singles_x_max = COURT_WIDTH - SINGLES_MARGIN
        net_y = NET_POSITION
    
    validated_hit_x = hit_x
    validated_hit_y = hit_y
    
    if court_rotation_90:
        if player_code == 'A':
            if hit_x < net_x + MIN_DISTANCE_FROM_NET:
                validated_hit_x = net_x + MIN_DISTANCE_FROM_NET
                warnings.append(f"Hit X clamped to stay on Player A's side of net")
        elif player_code == 'B':
            if hit_x > net_x - MIN_DISTANCE_FROM_NET:
                validated_hit_x = net_x - MIN_DISTANCE_FROM_NET
                warnings.append(f"Hit X clamped to stay on Player B's side of net")
        
        if hit_y < singles_y_min:
            validated_hit_y = singles_y_min
            warnings.append(f"Hit Y clamped to singles court boundary")
        elif hit_y > singles_y_max:
            validated_hit_y = singles_y_max
            warnings.append(f"Hit Y clamped to singles court boundary")
    else:
        if hit_x < singles_x_min:
            validated_hit_x = singles_x_min
            warnings.append(f"Hit X clamped to singles court boundary")
        elif hit_x > singles_x_max:
            validated_hit_x = singles_x_max
            warnings.append(f"Hit X clamped to singles court boundary")
        
        if player_code == 'A':
            if hit_y < net_y + MIN_DISTANCE_FROM_NET:
                validated_hit_y = net_y + MIN_DISTANCE_FROM_NET
                warnings.append(f"Hit Y clamped to stay on Player A's side of net")
        elif player_code == 'B':
            if hit_y > net_y - MIN_DISTANCE_FROM_NET:
                validated_hit_y = net_y - MIN_DISTANCE_FROM_NET
                warnings.append(f"Hit Y clamped to stay on Player B's side of net")
    
    validated_landing_x = landing_x
    validated_landing_y = landing_y
    
    if court_rotation_90:
        if player_code == 'A':
            if landing_x > net_x - MIN_DISTANCE_FROM_NET:
                validated_landing_x = net_x - MIN_DISTANCE_FROM_NET
                warnings.append(f"Landing X clamped to be beyond net (Player A's shot)")
        elif player_code == 'B':
            if landing_x < net_x + MIN_DISTANCE_FROM_NET:
                validated_landing_x = net_x + MIN_DISTANCE_FROM_NET
                warnings.append(f"Landing X clamped to be beyond net (Player B's shot)")
        
        if landing_y < 0.0:
            validated_landing_y = 0.0
            warnings.append(f"Landing Y clamped to court boundary")
        elif landing_y > court_y_max:
            validated_landing_y = court_y_max
            warnings.append(f"Landing Y clamped to court boundary")
    else:
        if player_code == 'A':
            if landing_y > net_y - MIN_DISTANCE_FROM_NET:
                validated_landing_y = net_y - MIN_DISTANCE_FROM_NET
                warnings.append(f"Landing Y clamped to be beyond net (Player A's shot)")
        elif player_code == 'B':
            if landing_y < net_y + MIN_DISTANCE_FROM_NET:
                validated_landing_y = net_y + MIN_DISTANCE_FROM_NET
                warnings.append(f"Landing Y clamped to be beyond net (Player B's shot)")
        
        if landing_x < 0.0:
            validated_landing_x = 0.0
            warnings.append(f"Landing X clamped to court boundary")
        elif landing_x > court_x_max:
            validated_landing_x = court_x_max
            warnings.append(f"Landing X clamped to court boundary")
    
    return validated_hit_x, validated_hit_y, validated_landing_x, validated_landing_y, warnings

def validate_coordinate_arrays(hit_x_list, hit_y_list, landing_x_list, landing_y_list, player_codes, court_rotation_90=False):
    """Validate arrays of shot coordinates"""
    validated_hit_x = []
    validated_hit_y = []
    validated_landing_x = []
    validated_landing_y = []
    total_warnings = []
    
    for i in range(len(hit_x_list)):
        player_code = player_codes[i] if i < len(player_codes) else 'A'
        
        v_hx, v_hy, v_lx, v_ly, warnings = validate_shot_coordinates(
            hit_x_list[i], hit_y_list[i], 
            landing_x_list[i], landing_y_list[i], 
            player_code, court_rotation_90
        )
        
        validated_hit_x.append(v_hx)
        validated_hit_y.append(v_hy)
        validated_landing_x.append(v_lx)
        validated_landing_y.append(v_ly)
        total_warnings.extend(warnings)
    
    return validated_hit_x, validated_hit_y, validated_landing_x, validated_landing_y, total_warnings

def get_all_players():
    df = load_data()
    return sorted(set(df['winner'].dropna()) | set(df['loser'].dropna()))

def get_all_tournaments():
    df = load_data()
    return sorted(df['tournament'].dropna().unique())

def get_opponents_for(player_name):
    df = load_data()
    opponents = set(df[df['winner'] == player_name]['loser'].dropna())
    opponents.update(df[df['loser'] == player_name]['winner'].dropna())
    return sorted(opponents)

def get_tournaments_for_match(player_name, opponent_name):
    df = load_data()
    match_tournaments = set()
    
    # Find tournaments where player vs opponent played
    player_wins = df[(df['winner'] == player_name) & (df['loser'] == opponent_name)]
    player_losses = df[(df['winner'] == opponent_name) & (df['loser'] == player_name)]
    
    match_tournaments.update(player_wins['tournament'].dropna().unique())
    match_tournaments.update(player_losses['tournament'].dropna().unique())
    
    return sorted(match_tournaments)

def get_rounds_for_match(player_name, opponent_name, tournament_name):
    df = load_data()
    match_rounds = set()
    
    # Find rounds where player vs opponent played in the specific tournament
    player_wins = df[(df['winner'] == player_name) & (df['loser'] == opponent_name) & 
                     (df['tournament'] == tournament_name)]
    player_losses = df[(df['winner'] == opponent_name) & (df['loser'] == player_name) & 
                       (df['tournament'] == tournament_name)]
    
    match_rounds.update(player_wins['round'].dropna().unique())
    match_rounds.update(player_losses['round'].dropna().unique())
    
    return sorted(match_rounds)

def get_dashboard_data(form):
    df = load_data()
    tournament = form.get('tournament')
    round_ = form.get('round')
    match = form.get('match')
    sets = form.getlist('sets')
    player_a, player_b = match.split(" vs ")
    selected_sets = [int(s) for s in sets]
    match_data = df[
        (df["tournament"] == tournament) &
        (df["round"] == round_) &
        (df["set_number"].isin(selected_sets)) &
        (
            ((df["winner"] == player_a) & (df["loser"] == player_b)) |
            ((df["winner"] == player_b) & (df["loser"] == player_a))
        )
    ]
    kpi_data = generate_kpi_analysis(match_data, player_a, player_b)
    # --- ADVANCED KPIs ---
    winrate_a = compute_win_rate_by_rally_length(match_data, player_a, player_a, player_b)
    winrate_b = compute_win_rate_by_rally_length(match_data, player_b, player_a, player_b)
    serve_receive_a = serve_receive_performance(match_data, player_a, player_a, player_b)
    serve_receive_b = serve_receive_performance(match_data, player_b, player_a, player_b)
    shot_effectiveness_a = shot_effectiveness(match_data, player_a, player_a, player_b)
    shot_effectiveness_b = shot_effectiveness(match_data, player_b, player_a, player_b)
    shot_freq_a = shot_selection_frequency(match_data, player_a, player_a, player_b)
    shot_freq_b = shot_selection_frequency(match_data, player_b, player_a, player_b)
    error_breakdown_a = error_type_breakdown(match_data, player_a, player_a, player_b)
    error_breakdown_b = error_type_breakdown(match_data, player_b, player_a, player_b)
    streaks_a = calculate_streaks_by_set(match_data, player_a, player_a, player_b)
    streaks_b = calculate_streaks_by_set(match_data, player_b, player_a, player_b)
    score_progression_df = get_score_progression(match_data, player_a, player_b)
    score_progression_json = score_progression_df.to_dict(orient="records")
    clutch_win_a = calculate_clutch_performance(match_data, player_a, player_a, player_b)
    clutch_win_b = calculate_clutch_performance(match_data, player_b, player_a, player_b)
    # For mean winning shots, use only selected set (if available)
    selected_set = selected_sets[0] if selected_sets else None
    if selected_set is not None:
        df_set = match_data[match_data["set_number"] == selected_set]
        mean_winning_shots_a = compute_mean_winning_shots(df_set, player_a, player_a, player_b)
        mean_winning_shots_b = compute_mean_winning_shots(df_set, player_b, player_a, player_b)
    else:
        mean_winning_shots_a = 0.0
        mean_winning_shots_b = 0.0
    return {
        "tournament": tournament,
        "player_a": player_a,
        "player_b": player_b,
        "df_badminton": match_data,  # Add the filtered dataframe
        "records": match_data.head(10).to_dict(orient='records'),
        "kpi_data": kpi_data,
        "winrate_a": winrate_a,
        "winrate_b": winrate_b,
        "serve_receive_a": serve_receive_a,
        "serve_receive_b": serve_receive_b,
        "shot_effectiveness_a": shot_effectiveness_a,
        "shot_effectiveness_b": shot_effectiveness_b,
        "shot_freq_a": shot_freq_a,
        "shot_freq_b": shot_freq_b,
        "error_breakdown_a": error_breakdown_a,
        "error_breakdown_b": error_breakdown_b,
        "streaks_a": streaks_a,
        "streaks_b": streaks_b,
        "score_progression": score_progression_json,
        "clutch_win_a": clutch_win_a,
        "clutch_win_b": clutch_win_b,
        "mean_winning_shots_a": mean_winning_shots_a,
        "mean_winning_shots_b": mean_winning_shots_b,
    }

def generate_kpi_analysis(df_badminton, player_a, player_b):
    # Check if we have the required columns
    required_columns = ['winner', 'loser', 'roundscore_A', 'roundscore_B', 'getpoint_player']
    if not all(col in df_badminton.columns for col in required_columns):
        # If columns don't exist, create dummy data for demonstration
        return {
            'mean_winner_points': 0,
            'mean_loser_points': 0,
            'histogram_chart': None,
            'distribution_chart': None,
            'draw_probability': 0,
            'model_data': None,
            'rally_analysis': None
        }
    
    # Calculate rally-based statistics
    rally_stats = calculate_rally_statistics(df_badminton, player_a, player_b)
    
    # Generate charts
    histogram_chart = generate_rally_histogram(df_badminton)
    distribution_chart = generate_rally_distribution(df_badminton)
    
    # Calculate draw probability (simplified)
    draw_probability = calculate_draw_probability(df_badminton, player_a, player_b)
    
    # Create model data for rally analysis
    model_data = create_rally_model_data(df_badminton)
    
    return {
        'mean_winner_points': rally_stats['mean_winner_rallies'],
        'mean_loser_points': rally_stats['mean_loser_rallies'],
        'histogram_chart': histogram_chart,
        'distribution_chart': distribution_chart,
        'draw_probability': draw_probability,
        'model_data': model_data,
        'rally_analysis': rally_stats
    }

def calculate_rally_statistics(df_badminton, player_a, player_b):
    # First try to use getpoint_player column
    if 'getpoint_player' in df_badminton.columns and df_badminton['getpoint_player'].notna().sum() > 0:
        # Count rallies won by each player using getpoint_player
        player_a_wins = df_badminton[df_badminton['getpoint_player'] == 'A'].shape[0]
        player_b_wins = df_badminton[df_badminton['getpoint_player'] == 'B'].shape[0]
    else:
        # Fallback: estimate based on score progression
        # Count how many times each player's score increased
        player_a_wins = 0
        player_b_wins = 0
        
        # Group by set and track score changes
        for set_num in df_badminton['set_number'].unique():
            set_data = df_badminton[df_badminton['set_number'] == set_num].sort_values('rally')
            if len(set_data) > 1:
                # Count score increases for each player
                for i in range(1, len(set_data)):
                    prev_score_a = set_data.iloc[i-1]['roundscore_A']
                    curr_score_a = set_data.iloc[i]['roundscore_A']
                    prev_score_b = set_data.iloc[i-1]['roundscore_B']
                    curr_score_b = set_data.iloc[i]['roundscore_B']
                    
                    if curr_score_a > prev_score_a:
                        player_a_wins += 1
                    if curr_score_b > prev_score_b:
                        player_b_wins += 1
    
    # Calculate total rallies
    total_rallies = player_a_wins + player_b_wins
    
    # Calculate win percentages
    player_a_win_rate = round((player_a_wins / total_rallies * 100), 2) if total_rallies > 0 else 0
    player_b_win_rate = round((player_b_wins / total_rallies * 100), 2) if total_rallies > 0 else 0
    
    # Calculate average rallies per set
    unique_sets = df_badminton['set_number'].nunique()
    avg_rallies_per_set = round(total_rallies / unique_sets, 2) if unique_sets > 0 else 0
    
    return {
        'player_a_rallies': player_a_wins,
        'player_b_rallies': player_b_wins,
        'total_rallies': total_rallies,
        'player_a_win_rate': player_a_win_rate,
        'player_b_win_rate': player_b_win_rate,
        'avg_rallies_per_set': avg_rallies_per_set,
        'mean_winner_rallies': player_a_wins,  # For compatibility with existing code
        'mean_loser_rallies': player_b_wins    # For compatibility with existing code
    }

def generate_rally_histogram(df_badminton):
    try:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        # Get rally scores - use roundscore_A and roundscore_B
        player_a_scores = df_badminton['roundscore_A'].values
        player_b_scores = df_badminton['roundscore_B'].values
        
        # Filter out zeros to focus on actual scoring
        player_a_scores = player_a_scores[player_a_scores > 0]
        player_b_scores = player_b_scores[player_b_scores > 0]
        
        # Create histogram
        ax.hist([player_a_scores, player_b_scores],
                bins=range(0, 25, 2),
                alpha=0.7,
                label=['Player A Scores', 'Player B Scores'],
                color=["#1a04d9", "#2c9623"])
        
        plt.xticks(range(0, 25, 2))
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution at Rally-End (per Player)')
        plt.legend()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    except Exception as e:
        print(f"Error generating rally histogram: {e}")
        return None

def generate_rally_distribution(df_badminton):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        
        # Get score data
        all_scores = np.concatenate([df_badminton['roundscore_A'].values, df_badminton['roundscore_B'].values])
        all_scores = all_scores[all_scores > 0]  # Filter out zeros
        
        # Calculate mean for Poisson distribution
        mu_score = np.mean(all_scores) if len(all_scores) > 0 else 1
        
        # Plot 1: Score Distribution
        ax1.hist(all_scores, bins=range(0, 25), density=True, alpha=0.7, color='#2c9623', label='Actual Scores')
        ax1.plot(range(0, 25), poisson.pmf(range(0, 25), mu_score), '-o', ms=5, color="#1a04d9", label=f'Poisson (μ={mu_score:.1f})')
        ax1.set_title('Score Distribution')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Plot 2: Player Performance (if getpoint_player available)
        if 'getpoint_player' in df_badminton.columns and df_badminton['getpoint_player'].notna().sum() > 0:
            player_performance = df_badminton['getpoint_player'].value_counts()
            ax2.bar(range(len(player_performance)), player_performance.values, color=['#1a04d9', '#2c9623'])
            ax2.set_title('Rallies Won by Player')
            ax2.set_xlabel('Player')
            ax2.set_ylabel('Rallies Won')
            ax2.set_xticks(range(len(player_performance)))
            ax2.set_xticklabels(['Player A', 'Player B'])
        else:
            # Fallback: show score comparison
            avg_score_a = df_badminton['roundscore_A'].mean()
            avg_score_b = df_badminton['roundscore_B'].mean()
            ax2.bar(['Player A', 'Player B'], [avg_score_a, avg_score_b], color=['#1a04d9', '#2c9623'])
            ax2.set_title('Average Score by Player')
            ax2.set_ylabel('Average Score')
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    except Exception as e:
        print(f"Error generating rally distribution: {e}")
        return None

def calculate_draw_probability(df_badminton, player_a, player_b):
    try:
        # Use getpoint_player if available
        if 'getpoint_player' in df_badminton.columns and df_badminton['getpoint_player'].notna().sum() > 0:
            player_a_rallies = df_badminton[df_badminton['getpoint_player'] == 'A'].shape[0]
            player_b_rallies = df_badminton[df_badminton['getpoint_player'] == 'B'].shape[0]
        else:
            # Fallback: use score-based estimation
            player_a_rallies = df_badminton['roundscore_A'].sum()
            player_b_rallies = df_badminton['roundscore_B'].sum()
        
        total_rallies = player_a_rallies + player_b_rallies
        
        if total_rallies == 0:
            return 0
        
        # Calculate probability of equal performance (simplified)
        p_a = player_a_rallies / total_rallies
        p_b = player_b_rallies / total_rallies
        
        # Simplified draw probability (when both players have similar performance)
        draw_prob = round(abs(p_a - p_b) * 0.1, 2)  # Simplified calculation
        
        return draw_prob
    except Exception as e:
        print(f"Error calculating draw probability: {e}")
        return 0

def create_rally_model_data(df_badminton):
    try:
        # Create a simplified model data structure
        rally_data = []
        
        # Use getpoint_player if available
        if 'getpoint_player' in df_badminton.columns and df_badminton['getpoint_player'].notna().sum() > 0:
            for _, row in df_badminton[df_badminton['getpoint_player'].notna()].head(20).iterrows():
                rally_data.append({
                    'player': 'Player A' if row['getpoint_player'] == 'A' else 'Player B',
                    'opponent': 'Player B' if row['getpoint_player'] == 'A' else 'Player A',
                    'points': row['roundscore_A'] if row['getpoint_player'] == 'A' else row['roundscore_B'],
                    'win': 1
                })
        else:
            # Fallback: create sample data based on scores
            for _, row in df_badminton.head(20).iterrows():
                if row['roundscore_A'] > 0:
                    rally_data.append({
                        'player': 'Player A',
                        'opponent': 'Player B',
                        'points': row['roundscore_A'],
                        'win': 1
                    })
                if row['roundscore_B'] > 0:
                    rally_data.append({
                        'player': 'Player B',
                        'opponent': 'Player A',
                        'points': row['roundscore_B'],
                        'win': 1
                    })
        
        return rally_data
    except Exception as e:
        print(f"Error creating rally model data: {e}")
        return []

def get_shot_landing_heatmap(player=None, opponent=None, tournament=None, round_name=None, set_number=None):
    df = load_data()
    
    if player and opponent:
        player_matches = (
            ((df['winner'] == player) & (df['loser'] == opponent)) |
            ((df['winner'] == opponent) & (df['loser'] == player))
        )
        df = df[player_matches]
    
    if tournament:
        df = df[df['tournament'] == tournament]
    
    if round_name:
        df = df[df['round'] == round_name]
    
    if set_number:
        try:
            set_number_int = int(set_number)
            df = df.copy()
            if 'set_number' in df.columns and df['set_number'].notna().any():
                df['set_number'] = pd.to_numeric(df['set_number'], errors='coerce').astype('Int64')
                df = df[df['set_number'] == set_number_int]
            elif 'set' in df.columns:
                df['set'] = pd.to_numeric(df['set'], errors='coerce').astype('Int64')
                df = df[df['set'] == set_number_int]
        except Exception as e:
            print(f"Invalid set_number: {set_number}")
    
    point_shots = df[df['win_reason'].notna() | df['lose_reason'].notna()]
    
    if 'id' in point_shots.columns and 'ball_round' in point_shots.columns and 'rally' in point_shots.columns:
        last_shots = point_shots.sort_values(['id', 'rally', 'ball_round']).groupby(['id', 'rally']).tail(1)
        point_shots = last_shots
    elif 'ball_round' in point_shots.columns and 'rally' in point_shots.columns:
        last_shots = point_shots.sort_values(['rally', 'ball_round']).groupby('rally').tail(1)
        point_shots = last_shots
    
    point_shots = point_shots[point_shots['landing_x'].notna() & point_shots['landing_y'].notna()]
    if point_shots.empty:
        return {'landing_x': [], 'landing_y': [], 'win_reason': []}

    landing_x_raw = point_shots['landing_x'].astype(float).tolist()
    landing_y_raw = point_shots['landing_y'].astype(float).tolist()

    max_x = max(landing_x_raw) if landing_x_raw else 0
    max_y = max(landing_y_raw) if landing_y_raw else 0
    needs_scaling = max_x > 20 or max_y > 20

    if needs_scaling:
        court_length = COURT_LENGTH
        court_width = COURT_WIDTH
        min_x, max_x = min(landing_x_raw), max(landing_x_raw)
        min_y, max_y = min(landing_y_raw), max(landing_y_raw)
        
        def scale(val, minv, maxv, target):
            return (val - minv) * (target / (maxv - minv)) if maxv != minv else target / 2
        
        # Apply 90-degree rotation: Y becomes X (length), X becomes Y (width)
        landing_x_scaled = [scale(y, min_y, max_y, court_length) for y in landing_y_raw]
        landing_y_scaled = [scale(x, min_x, max_x, court_width) for x in landing_x_raw]
        landing_x_list = landing_x_scaled
        landing_y_list = landing_y_scaled
    else:
        landing_x_list = landing_x_raw
        landing_y_list = landing_y_raw

    reason_map = {
        '掛網': 'Net',
        '出界': 'Out',
        '未過網': 'Did not cross net',
        '殺球得分': 'Smash Winner',
        '挑球得分': 'Lift Winner',
        '對方失誤': 'Opponent Error',
        '發球得分': 'Service Winner',
        '接發失誤': 'Return Error',
        'N/A': 'Unknown',
        '': 'Unknown',
        '對手未過網': 'Opponent did not cross net',
        '對手出界': 'Opponent out',
        '落地致勝': 'Winner (shuttle landed)',
        '對手掛網': 'Opponent net',
    }
    
    def map_reason(r):
        r_clean = r.strip() if isinstance(r, str) else r
        return reason_map.get(r_clean, r_clean if isinstance(r_clean, str) and r_clean.isascii() else 'Other')
    
    heatmap_data = {
        'landing_x': landing_x_list,
        'landing_y': landing_y_list,
        'win_reason': [
            map_reason(row['win_reason']) if pd.notna(row['win_reason']) else
            map_reason(row['lose_reason'])
            for _, row in point_shots.iterrows()
        ]
    }

    SINGLES_Y_MIN = SINGLES_MARGIN
    SINGLES_Y_MAX = COURT_WIDTH - SINGLES_MARGIN
    COURT_X_MIN = 0.0
    COURT_X_MAX = COURT_LENGTH
    NET_X = COURT_LENGTH / 2
    NUDGE = 0.16
    NUDGE_OUT = 0.05
    RAND_OFFSET = 0.02
    
    for i, reason in enumerate(heatmap_data['win_reason']):
        if reason == 'Opponent out':
            x = heatmap_data['landing_x'][i]
            y = heatmap_data['landing_y'][i]
            dist_left = abs(y - SINGLES_Y_MIN)
            dist_right = abs(y - SINGLES_Y_MAX)
            dist_back = abs(x - COURT_X_MAX)
            dist_front = abs(x - COURT_X_MIN)
            min_dist = min(dist_left, dist_right, dist_back, dist_front)
            
            if min_dist == dist_left:
                heatmap_data['landing_y'][i] = SINGLES_Y_MIN - NUDGE_OUT
                heatmap_data['landing_x'][i] = min(max(x + random.uniform(-RAND_OFFSET, RAND_OFFSET), COURT_X_MIN), COURT_X_MAX)
            elif min_dist == dist_right:
                heatmap_data['landing_y'][i] = SINGLES_Y_MAX + NUDGE_OUT
                heatmap_data['landing_x'][i] = min(max(x + random.uniform(-RAND_OFFSET, RAND_OFFSET), COURT_X_MIN), COURT_X_MAX)
            elif min_dist == dist_back:
                heatmap_data['landing_x'][i] = COURT_X_MAX + NUDGE_OUT
                heatmap_data['landing_y'][i] = min(max(y + random.uniform(-RAND_OFFSET, RAND_OFFSET), SINGLES_Y_MIN), SINGLES_Y_MAX)
            else:
                heatmap_data['landing_x'][i] = COURT_X_MIN - NUDGE_OUT
                heatmap_data['landing_y'][i] = min(max(y + random.uniform(-RAND_OFFSET, RAND_OFFSET), SINGLES_Y_MIN), SINGLES_Y_MAX)
        elif reason == 'Opponent net':
            if heatmap_data['landing_x'][i] < NET_X:
                heatmap_data['landing_x'][i] = NET_X - NUDGE
            else:
                heatmap_data['landing_x'][i] = NET_X + NUDGE

    return heatmap_data

def get_serve_return_placement(player=None, serve_type=None, winner=None, loser=None):
    """
    Get serve and return placement data for visualization.
    Uses landing_x, landing_y coordinates to show where serves/returns land.
    Now uses the same 90-degree rotated coordinate system as shot type spatial signature.
    Removed mode filter - now shows all serve and return types together.
    """
    df = load_data()
    if winner and loser:
        df = df[((df['winner'] == winner) & (df['loser'] == loser)) |
                ((df['winner'] == loser) & (df['loser'] == winner))]
    
    # Filter by serve type if specified
    if serve_type and serve_type != '':
        # Create reverse mapping from English back to original names
        reverse_serve_type_map = {
            'Unknown Shot Type': '未知球種',
            'Short Service': 'short service',
            'Long Service': 'long service',
            'Defensive Return Drive': 'defensive return drive',
            'Defensive Return Lob': 'defensive return lob',
            'Return Net': 'return net'
        }
        original_serve_type = reverse_serve_type_map.get(serve_type, serve_type)
        df = df[df['type'] == original_serve_type]
    
    if player and player != '':
        player_code = None
        if winner and player == winner:
            player_code = 'A'
        elif loser and player == loser:
            player_code = 'B'
        else:
            player_code = player
        
        df = df[df['player'] == player_code]
    
    serve_return_types = ['short service', 'long service', 'defensive return drive', 'defensive return lob', 'return net', '未知球種']
    filtered = df[df['type'].isin(serve_return_types)]
    
    filtered = filtered[filtered['landing_x'].notna() & filtered['landing_y'].notna()]
    
    if len(filtered) > 0:
        # Use the same scaling and rotation as shot type spatial signature
        court_width = COURT_WIDTH    # 6.1m
        court_length = COURT_LENGTH  # 13.4m
        # Use min/max of both axes for scaling (landing only, since hit is not used)
        all_x_min = filtered['landing_x'].min()
        all_x_max = filtered['landing_x'].max()
        all_y_min = filtered['landing_y'].min()
        all_y_max = filtered['landing_y'].max()
        # Add padding
        x_padding = (all_x_max - all_x_min) * 0.05
        y_padding = (all_y_max - all_y_min) * 0.10
        effective_x_min = all_x_min - x_padding
        effective_x_max = all_x_max + x_padding
        effective_y_min = all_y_min - y_padding
        effective_y_max = all_y_max + y_padding
        # ROTATED: original Y (length) -> X, original X (width) -> Y
        if effective_y_max != effective_y_min:
            scaled_landing_x = filtered['landing_y'].apply(lambda val: (val - effective_y_min) * (court_length / (effective_y_max - effective_y_min)))
        else:
            scaled_landing_x = [court_length / 2] * len(filtered)
        if effective_x_max != effective_x_min:
            scaled_landing_y = filtered['landing_x'].apply(lambda val: (val - effective_x_min) * (court_width / (effective_x_max - effective_x_min)))
        else:
            scaled_landing_y = [court_width / 2] * len(filtered)
        # Dummy hit coordinates (center of court)
        hit_x_list = [court_length / 2] * len(filtered)
        hit_y_list = [court_width / 2] * len(filtered)
        landing_x_list = scaled_landing_x.tolist() if hasattr(scaled_landing_x, 'tolist') else list(scaled_landing_x)
        landing_y_list = scaled_landing_y.tolist() if hasattr(scaled_landing_y, 'tolist') else list(scaled_landing_y)
        player_codes = filtered['player'].tolist() if 'player' in filtered.columns else ['A'] * len(landing_x_list)
        # Validate in rotated system
        _, _, validated_landing_x, validated_landing_y, validation_warnings = validate_coordinate_arrays(
            hit_x_list, hit_y_list, landing_x_list, landing_y_list, player_codes, court_rotation_90=True
        )
        if validation_warnings:
            unique_warnings = list(set(validation_warnings))
            for warning in unique_warnings:
                count = validation_warnings.count(warning)
                print(f"[VALIDATION] {count} serve/return positions: {warning}")
        
        # Translate Chinese serve types to English
        serve_type_map = {
            '未知球種': 'Unknown Shot Type',
            'short service': 'Short Service',
            'long service': 'Long Service',
            'defensive return drive': 'Defensive Return Drive',
            'defensive return lob': 'Defensive Return Lob',
            'return net': 'Return Net'
        }
        
        def translate_serve_type(serve_type):
            return serve_type_map.get(serve_type, serve_type)
        
        translated_types = [translate_serve_type(t) for t in filtered['type'].tolist()]
        
        return {
            'hit_x': hit_x_list,
            'hit_y': hit_y_list,
            'landing_x': [float(x) for x in validated_landing_x],
            'landing_y': [float(y) for y in validated_landing_y],
            'type': translated_types,
            'player': filtered['player'].tolist(),
            'server': filtered['server'].tolist(),
        }
    else:
        return {
            'hit_x': [],
            'hit_y': [],
            'landing_x': [],
            'landing_y': [],
            'type': [],
            'player': [],
            'server': [],
        }

def get_available_serve_types(winner=None, loser=None, player=None):
    """
    Get available serve types for the current match context.
    Removed mode filter - now returns all serve and return types.
    """
    df = load_data()
    
    if winner and loser:
        df = df[((df['winner'] == winner) & (df['loser'] == loser)) | 
                ((df['winner'] == loser) & (df['loser'] == winner))]
    
    serve_return_types = ['short service', 'long service', 'defensive return drive', 'defensive return lob', 'return net', '未知球種']
    filtered = df[df['type'].isin(serve_return_types)]
    
    if player:
        if player == winner:
            serve_filtered = filtered[filtered['server'] == 1]
            return_filtered = filtered[(filtered['ball_round'] == 2) & (filtered['player'] == 'A')]
            filtered = pd.concat([serve_filtered, return_filtered])
        elif player == loser:
            serve_filtered = filtered[filtered['server'] == 2]
            return_filtered = filtered[(filtered['ball_round'] == 2) & (filtered['player'] == 'B')]
            filtered = pd.concat([serve_filtered, return_filtered])
    
    serve_types_raw = sorted(filtered['type'].dropna().unique().tolist())
    
    serve_type_map = {
        '未知球種': 'Unknown Shot Type',
        'short service': 'Short Service',
        'long service': 'Long Service',
        'defensive return drive': 'Defensive Return Drive',
        'defensive return lob': 'Defensive Return Lob',
        'return net': 'Return Net'
    }
    
    serve_types = [serve_type_map.get(st, st) for st in serve_types_raw]
    return serve_types

def get_error_zone_data(winner=None, loser=None, player=None, error_type=None, set_number=None):
    """
    Get error zone data for visualization - shows where unforced errors occur.
    Uses hit_x, hit_y coordinates to show where the error was made.
    Now uses the same 90-degree rotated coordinate system as shot type spatial signature.
    """
    df = load_data()
    if winner and loser:
        df = df[((df['winner'] == winner) & (df['loser'] == loser)) |
                ((df['winner'] == loser) & (df['loser'] == winner))]
    
    if set_number:
        try:
            set_number_int = int(set_number)
            df = df.copy()
            df['set_number'] = pd.to_numeric(df['set_number'], errors='coerce').astype('Int64')
            df = df[df['set_number'] == set_number_int]
        except Exception as e:
            print(f"Invalid set_number: {set_number}")
    
    if player:
        player_code = None
        if winner and player == winner:
            player_code = 'A'
        elif loser and player == loser:
            player_code = 'B'
        else:
            player_code = player
        
        df = df[df['player'] == player_code]
    
    unforced_error_reasons = ['掛網', '出界', '未過網', '落點判斷失誤', 'error', 'net', 'out']
    player_errors_df = df[df['lose_reason'].isin(unforced_error_reasons)]
    
    if error_type and error_type != '':
        player_errors_df = player_errors_df[player_errors_df['lose_reason'] == error_type]
    
    # Count errors by type for the result
    error_counts = {}
    if not player_errors_df.empty:
        error_counts = player_errors_df['lose_reason'].value_counts().to_dict()
    
    result = {
        'hit_x': [],
        'hit_y': [],
        'player': [],
        'error_type': [],
        'error_counts': error_counts
    }
    
    # If there are errors for the specific selection, calculate and add coordinates
    if not player_errors_df.empty:
        # Use the same scaling and rotation as shot type spatial signature
        court_width = COURT_WIDTH    # 6.1m
        court_length = COURT_LENGTH  # 13.4m
        
        # Get min/max for scaling
        all_x_min = player_errors_df['hit_x'].min()
        all_x_max = player_errors_df['hit_x'].max()
        all_y_min = player_errors_df['hit_y'].min()
        all_y_max = player_errors_df['hit_y'].max()
        
        # Add padding to ensure errors don't go right to the edges
        x_padding = (all_x_max - all_x_min) * 0.05  # 5% padding
        y_padding = (all_y_max - all_y_min) * 0.10  # 10% padding for Y to reduce gaps
        effective_x_min = all_x_min - x_padding
        effective_x_max = all_x_max + x_padding
        effective_y_min = all_y_min - y_padding
        effective_y_max = all_y_max + y_padding
        
        # ROTATED: original Y (length) -> X, original X (width) -> Y
        if effective_y_max != effective_y_min:
            scaled_hit_x = player_errors_df['hit_y'].apply(lambda val: (val - effective_y_min) * (court_length / (effective_y_max - effective_y_min)))
        else:
            scaled_hit_x = [court_length / 2] * len(player_errors_df)
        if effective_x_max != effective_x_min:
            scaled_hit_y = player_errors_df['hit_x'].apply(lambda val: (val - effective_x_min) * (court_width / (effective_x_max - effective_x_min)))
        else:
            scaled_hit_y = [court_width / 2] * len(player_errors_df)
        
        # Apply nudging rules for specific error types
        net_line_x = court_length / 2  # Net is at the middle of the court
        singles_margin = 0.46  # Distance from doubles line to singles line
        singles_width = 5.18   # Singles court width
        outside_nudge_distance = 0.05  # 5cm outside the singles line
        
        # Create lists for processing
        hit_x_list = scaled_hit_x.tolist() if hasattr(scaled_hit_x, 'tolist') else list(scaled_hit_x)
        hit_y_list = scaled_hit_y.tolist() if hasattr(scaled_hit_y, 'tolist') else list(scaled_hit_y)
        error_types = player_errors_df['lose_reason'].tolist()
        
        # Apply nudging based on error type for visualization
        for i, error_type in enumerate(error_types):
            if error_type in ['掛網', 'net']:  # Net errors
                # Nudge to net line with small random offset
                import random
                random_offset = random.uniform(-0.2, 0.2)  # ±20cm along net
                hit_x_list[i] = net_line_x + random_offset
                # Keep Y position as is, but ensure it's within court bounds
                hit_y_list[i] = max(0, min(court_width, hit_y_list[i]))
                
            elif error_type in ['出界', 'out']:  # Out errors
                # Nudge to just outside singles boundary
                current_y = hit_y_list[i]
                if current_y < court_width / 2:  # Left side of court
                    # Nudge outside left singles line
                    hit_y_list[i] = singles_margin - outside_nudge_distance
                else:  # Right side of court
                    # Nudge outside right singles line
                    hit_y_list[i] = singles_margin + singles_width + outside_nudge_distance
                # Add small random offset for natural appearance
                import random
                random_offset = random.uniform(-0.1, 0.1)  # ±10cm
                hit_y_list[i] += random_offset
        
        # Create dummy landing coordinates (not used for errors, but required by validation)
        landing_x_list = [court_length / 2] * len(hit_x_list)
        landing_y_list = [court_width / 2] * len(hit_y_list)
        
        # Get player codes for validation
        player_codes = player_errors_df['player'].tolist() if 'player' in player_errors_df.columns else ['A'] * len(hit_x_list)
        
        # Apply coordinate validation (rotated system)
        validated_hit_x, validated_hit_y, _, _, validation_warnings = validate_coordinate_arrays(
            hit_x_list, hit_y_list, landing_x_list, landing_y_list, player_codes, court_rotation_90=True
        )
        
        # Log validation warnings
        if validation_warnings:
            unique_warnings = list(set(validation_warnings))
            for warning in unique_warnings:
                count = validation_warnings.count(warning)
                print(f"[VALIDATION] {count} error positions: {warning}")
        
        # Translate Chinese error types to English for display
        error_type_map = {
            '掛網': 'Net',
            '出界': 'Out',
            '未過網': 'Did not cross net',
            '落點判斷失誤': 'Landing misjudgment',
            'error': 'Error',
            'net': 'Net',
            'out': 'Out'
        }
        
        def translate_error_type(error_type):
            return error_type_map.get(error_type, error_type)
        
        translated_error_types = [translate_error_type(t) for t in player_errors_df['lose_reason'].tolist()]
        
        result['hit_x'] = [float(x) for x in validated_hit_x]
        result['hit_y'] = [float(y) for y in validated_hit_y]
        result['player'] = player_errors_df['player'].tolist()
        result['error_type'] = translated_error_types
    else:
        print(" get_error_zone_data - Returning empty plot data but with error counts.")
        
    return result

def get_available_error_types(winner=None, loser=None, player=None, tournament=None, round_name=None, set_number=None):
    """
    Return the unique error types (lose_reason) for the current match context (and optionally player, tournament, round).
    """
    df = load_data()
    # Filter to the current match context
    if winner and loser:
        df = df[((df['winner'] == winner) & (df['loser'] == loser)) |
                ((df['winner'] == loser) & (df['loser'] == winner))]
    if tournament:
        df = df[df['tournament'] == tournament]
    if round_name:
        df = df[df['round'] == round_name]
        
    # Filter by set if provided
    if set_number and set_number != '':
        try:
            set_number_int = int(set_number)
            df = df[df['set_number'] == set_number_int]
        except (ValueError, TypeError):
            print(f"[ERROR] Invalid set number for available error types: {set_number}")

    # Only look at error rows
    df['lose_reason'] = df['lose_reason'].astype(str).str.strip().str.lower()
    unforced_error_reasons = ['error', 'out', 'net', '掛網', '出界', '未過網']
    error_df = df[df['lose_reason'].isin(unforced_error_reasons)]
    # Optionally filter by player (map to 'A' or 'B')
    if player and player != '':
        if winner and player == winner:
            player_code = 'A'
        elif loser and player == loser:
            player_code = 'B'
        else:
            player_code = player
        error_df['player'] = error_df['player'].astype(str).str.strip()
        error_df = error_df[error_df['player'] == player_code]
    # Return unique error types present in this context
    error_types = sorted(error_df['lose_reason'].unique())
    return error_types

def get_movement_density_heatmap(winner=None, loser=None, player=None, round_name=None, set_number=None):
    """
    Generate movement density heatmap data showing where players spend most time.
    Uses hit_x, hit_y coordinates to show player positions during shots.
    Uses the same 90-degree rotated coordinate system as shot type spatial signature.
    Movement positions are clamped to the full doubles court boundaries (not singles).
    """
    df = load_data()

    
    # Filter to the current match context
    if winner and loser:
        df = df[((df['winner'] == winner) & (df['loser'] == loser)) |
                ((df['winner'] == loser) & (df['loser'] == winner))]

    
    # Filter by player if specified
    if player:
        # Map player name to 'A' or 'B' for this match
        player_code = None
        if winner and player == winner:
            player_code = 'A'
        elif loser and player == loser:
            player_code = 'B'
        else:
            player_code = player
        
        df = df[df['player'] == player_code]

    
    # Filter by round if specified
    if round_name:
        df = df[df['round'] == round_name]

    
    # Filter by set if specified
    if set_number and set_number != '':
        try:
            set_number_int = int(set_number)
            df = df[df['set_number'] == set_number_int]

        except (ValueError, TypeError):
            print(f"[ERROR] Invalid set number for movement density: {set_number}")
    
    # Only keep rows with valid hit coordinates
    df = df[df['hit_x'].notna() & df['hit_y'].notna()]

    
    if df.empty:
        print(" No movement data available")
        return {'x': [], 'y': []}
    
    # Use the same scaling and rotation as shot type spatial signature
    court_width = COURT_WIDTH    # 6.1m
    court_length = COURT_LENGTH  # 13.4m
    
    # Get min/max for scaling
    all_x_min = df['hit_x'].min()
    all_x_max = df['hit_x'].max()
    all_y_min = df['hit_y'].min()
    all_y_max = df['hit_y'].max()
    
    # Add padding to ensure positions don't go right to the edges
    x_padding = (all_x_max - all_x_min) * 0.05  # 5% padding
    y_padding = (all_y_max - all_y_min) * 0.10  # 10% padding for Y to reduce gaps
    effective_x_min = all_x_min - x_padding
    effective_x_max = all_x_max + x_padding
    effective_y_min = all_y_min - y_padding
    effective_y_max = all_y_max + y_padding
    
    # Apply 90-degree rotation: Y becomes X (length), X becomes Y (width)
    if effective_y_max != effective_y_min:
        scaled_hit_x = df['hit_y'].apply(lambda val: (val - effective_y_min) * (court_length / (effective_y_max - effective_y_min)))
    else:
        scaled_hit_x = [court_length / 2] * len(df)
    if effective_x_max != effective_x_min:
        scaled_hit_y = df['hit_x'].apply(lambda val: (val - effective_x_min) * (court_width / (effective_x_max - effective_x_min)))
    else:
        scaled_hit_y = [court_width / 2] * len(df)
    
    # Convert to lists for validation
    hit_x_list = scaled_hit_x.tolist() if hasattr(scaled_hit_x, 'tolist') else list(scaled_hit_x)
    hit_y_list = scaled_hit_y.tolist() if hasattr(scaled_hit_y, 'tolist') else list(scaled_hit_y)
    
    # Create dummy landing coordinates (not used for movement, but required by validation)
    landing_x_list = [court_length / 2] * len(hit_x_list)
    landing_y_list = [court_width / 2] * len(hit_y_list)
    
    # Get player codes for validation
    player_codes = df['player'].tolist() if 'player' in df.columns else ['A'] * len(hit_x_list)
    
    # Apply coordinate validation (rotated system)
    validated_hit_x, validated_hit_y, _, _, validation_warnings = validate_coordinate_arrays(
        hit_x_list, hit_y_list, landing_x_list, landing_y_list, player_codes, court_rotation_90=True
    )
    
    # Log validation warnings
    if validation_warnings:
        unique_warnings = list(set(validation_warnings))
        for warning in unique_warnings:
            count = validation_warnings.count(warning)
            print(f"[VALIDATION] {count} movement positions: {warning}")
    
    return {
        'x': [float(x) for x in validated_hit_x],
        'y': [float(y) for y in validated_hit_y]
    }

def get_shot_type_spatial_signature(winner=None, loser=None, player=None, shot_type=None, set_number=None):
    """
    Returns shot trajectory data showing where shots originate from and land.
    Includes hit_x/y (origin) and landing_x/y (destination) for visualization.
    """
    df = load_data()

    
    # Filter to the current match context
    if winner and loser:
        df = df[((df['winner'] == winner) & (df['loser'] == loser)) |
                ((df['winner'] == loser) & (df['loser'] == winner))]

    
    # Use set_number for set filtering
    if set_number:

        try:
            set_number_int = int(set_number)
            df = df[df['set_number'] == set_number_int]

        except Exception as e:
            print(f"Invalid set_number: {set_number}")
    
    # Filter by player if specified
    if player:
        # Map player name to 'A' or 'B' for this match
        player_code = None
        if winner and player == winner:
            player_code = 'A'
        elif loser and player == loser:
            player_code = 'B'
        else:
            player_code = player
        
        df = df[df['player'] == player_code]

    
    # Filter by shot type if specified
    if shot_type and shot_type != '':
        df = df[df['type'] == shot_type]

    
    # Only keep rows with valid hit and landing coordinates
    df = df[df['hit_x'].notna() & df['hit_y'].notna() & 
            df['landing_x'].notna() & df['landing_y'].notna()]

    
    if df.empty:
        print(" No shot trajectory data found after all filters")
        return {
            'hit_x': [], 'hit_y': [], 
            'landing_x': [], 'landing_y': [],
            'shot_types': [], 'rally_length': []
        }
    

    
    # Scale coordinates to court dimensions (90-degree rotation: X becomes Y, Y becomes X)
    court_width = COURT_WIDTH    # Court width in meters (now Y-axis)
    court_length = COURT_LENGTH  # Court length in meters (now X-axis)
    
    # Get combined min/max for each axis from ALL data (hit + landing)
    all_x_min = min(df['hit_x'].min(), df['landing_x'].min())
    all_x_max = max(df['hit_x'].max(), df['landing_x'].max())
    all_y_min = min(df['hit_y'].min(), df['landing_y'].min())
    all_y_max = max(df['hit_y'].max(), df['landing_y'].max())
    


    # Add padding to ensure shots don't go right to the edges
    x_padding = (all_x_max - all_x_min) * 0.05  # 5% padding
    y_padding = (all_y_max - all_y_min) * 0.10  # 10% padding for Y to reduce gaps
    
    effective_x_min = all_x_min - x_padding
    effective_x_max = all_x_max + x_padding
    effective_y_min = all_y_min - y_padding
    effective_y_max = all_y_max + y_padding

    # PROPER 90-DEGREE ROTATION: Swap X and Y coordinates
    # Original X (width) becomes new Y (width)
    # Original Y (length) becomes new X (length)
    
    # Scale original Y coordinates to court length (X-axis after rotation)
    if effective_y_max != effective_y_min:
        scaled_hit_x = df['hit_y'].apply(lambda val: (val - effective_y_min) * (court_length / (effective_y_max - effective_y_min)))
        scaled_landing_x = df['landing_y'].apply(lambda val: (val - effective_y_min) * (court_length / (effective_y_max - effective_y_min)))
    else:
        scaled_hit_x = [court_length / 2] * len(df)
        scaled_landing_x = [court_length / 2] * len(df)
        
    # Scale original X coordinates to court width (Y-axis after rotation)
    if effective_x_max != effective_x_min:
        scaled_hit_y = df['hit_x'].apply(lambda val: (val - effective_x_min) * (court_width / (effective_x_max - effective_x_min)))
        scaled_landing_y = df['landing_x'].apply(lambda val: (val - effective_x_min) * (court_width / (effective_x_max - effective_x_min)))
    else:
        scaled_hit_x = [court_width / 2] * len(df)
        scaled_landing_y = [court_width / 2] * len(df)
    
    # Convert to lists for validation
    hit_x_list = scaled_hit_x.tolist() if hasattr(scaled_hit_x, 'tolist') else list(scaled_hit_x)
    hit_y_list = scaled_hit_y.tolist() if hasattr(scaled_hit_y, 'tolist') else list(scaled_hit_y)
    landing_x_list = scaled_landing_x.tolist() if hasattr(scaled_landing_x, 'tolist') else list(scaled_landing_x)
    landing_y_list = scaled_landing_y.tolist() if hasattr(scaled_landing_y, 'tolist') else list(scaled_landing_y)
    
    # Get player codes for each shot
    player_codes = df['player'].tolist() if 'player' in df.columns else ['A'] * len(df)
    
    # Apply comprehensive coordinate validation (ROTATED system)
    validated_hit_x, validated_hit_y, validated_landing_x, validated_landing_y, validation_warnings = validate_coordinate_arrays(
        hit_x_list, hit_y_list, landing_x_list, landing_y_list, player_codes, court_rotation_90=True
    )
    
    # Log validation warnings
    if validation_warnings:
        unique_warnings = list(set(validation_warnings))
        for warning in unique_warnings:
            count = validation_warnings.count(warning)
            print(f"[VALIDATION] {count} shots: {warning}")
    
    # Count how many coordinates were modified
    modified_count = 0
    for i in range(len(hit_x_list)):
        if (hit_x_list[i] != validated_hit_x[i] or hit_y_list[i] != validated_hit_y[i] or 
            landing_x_list[i] != validated_landing_x[i] or landing_y_list[i] != validated_landing_y[i]):
            modified_count += 1
    
    if modified_count > 0:
        print(f"[VALIDATION] {modified_count} shots had coordinates validated to enforce court rules")
        


    # Return with correct mapping (ROTATED: X is length, Y is width)
    return {
        'hit_x': [float(x) for x in validated_hit_x],
        'hit_y': [float(y) for y in validated_hit_y],
        'landing_x': [float(x) for x in validated_landing_x],
        'landing_y': [float(y) for y in validated_landing_y],
        'shot_types': df['type'].tolist() if 'type' in df.columns else [],
        'rally_length': df['rally'].tolist() if 'rally' in df.columns else []
    }

def get_available_shot_types(winner=None, loser=None, player=None, set_number=None):
    """
    Return the unique shot types available for the current match context.
    """
    df = load_data()
    print(f" Getting available shot types - Initial rows: {len(df)}")
    
    # Filter to the current match context
    if winner and loser:
        df = df[((df['winner'] == winner) & (df['loser'] == loser)) |
                ((df['winner'] == loser) & (df['loser'] == winner))]

    
    # Use set_number for set filtering
    if set_number:
        try:
            set_number_int = int(set_number)
            df = df[df['set_number'] == set_number_int]
            print(f" After set_number filter: {len(df)} rows")
        except Exception as e:
            print(f" Invalid set_number: {set_number}")
    
    # Filter by player if specified
    if player:
        # Map player name to 'A' or 'B' for this match
        player_code = None
        if winner and player == winner:
            player_code = 'A'
        elif loser and player == loser:
            player_code = 'B'
        else:
            player_code = player
        
        df = df[df['player'] == player_code]

    
    # Only keep rows with valid shot type
    df = df[df['type'].notna()]
    print(f" After type filter: {len(df)} rows")
    
    # Return unique shot types
    shot_types = sorted(df['type'].unique())
    print(f" Available shot types: {shot_types}")
    return shot_types

def get_rally_end_zone_data(winner=None, loser=None, set_number=None):
    """
    Get rally end zone data - shows where rallies end and their length.
    Uses landing_x, landing_y coordinates for rally end positions.
    Outputs coordinates in the same 90-degree rotated system as shot type spatial signature.
    Translates win_reason from Chinese to English.
    """
    df = load_data()
    print(f" Rally End Zone Data - Initial rows: {len(df)}")
    
    # Filter to the current match context
    if winner and loser:
        df = df[((df['winner'] == winner) & (df['loser'] == loser)) |
                ((df['winner'] == loser) & (df['loser'] == winner))]

    
    # Use set_number for set filtering
    if set_number:
        try:
            set_number_int = int(set_number)
            df = df[df['set_number'] == set_number_int]
            print(f" After set_number filter: {len(df)} rows")
        except Exception as e:
            print(f" Invalid set_number: {set_number}")
    
    # Filter for rally ends (points where win_reason or lose_reason is not null)
    rally_ends_df = df[(df['win_reason'].notna()) | (df['lose_reason'].notna())]
    print(f" After rally end filter: {len(rally_ends_df)} rows")
    
    # Filter for last shot per rally (ball_round max per rally) - CRITICAL FOR ACCURACY
    if 'ball_round' in rally_ends_df.columns and 'rally' in rally_ends_df.columns:
        # Get the maximum ball_round per rally to identify the last shot
        last_shots = rally_ends_df.sort_values(['rally', 'ball_round']).groupby('rally').tail(1)
        rally_ends_df = last_shots
        print(f" After last shot per rally filter: {len(rally_ends_df)} rows")
    else:
        print("Warning: ball_round or rally columns not found, using all rally end shots")
    
    # Only keep rows with valid landing coordinates
    rally_ends_df = rally_ends_df[rally_ends_df['landing_x'].notna() & rally_ends_df['landing_y'].notna()]
    print(f" After coordinate filter: {len(rally_ends_df)} rows")
    
    if rally_ends_df.empty:
        print(" No rally end data found after all filters")
        return {
            'landing_x': [], 'landing_y': [], 
            'rally_length': [], 'win_reason': [], 'set_number': []
        }

    # 90-degree rotated system: X = court length, Y = court width
    court_width = COURT_WIDTH
    court_length = COURT_LENGTH

    # Get min/max for scaling
    all_x_min = rally_ends_df['landing_x'].min()
    all_x_max = rally_ends_df['landing_x'].max()
    all_y_min = rally_ends_df['landing_y'].min()
    all_y_max = rally_ends_df['landing_y'].max()

    # Scale original Y (length) to X (court length)
    if all_y_max != all_y_min:
        scaled_x = rally_ends_df['landing_y'].apply(lambda val: (val - all_y_min) * (court_length / (all_y_max - all_y_min)))
    else:
        scaled_x = [court_length / 2] * len(rally_ends_df)
    # Scale original X (width) to Y (court width)
    if all_x_max != all_x_min:
        scaled_y = rally_ends_df['landing_x'].apply(lambda val: (val - all_x_min) * (court_width / (all_x_max - all_x_min)))
    else:
        scaled_y = [court_width / 2] * len(rally_ends_df)

    # Convert to lists for validation
    landing_x_list = scaled_x.tolist()
    landing_y_list = scaled_y.tolist()
    # Create dummy hit coordinates (not used for landing positions, but required by validation)
    hit_x_list = [court_width / 2] * len(landing_x_list)
    hit_y_list = [court_length / 2] * len(landing_y_list)
    # Get player codes for validation
    player_codes = rally_ends_df['player'].tolist() if 'player' in rally_ends_df.columns else ['A'] * len(landing_x_list)
    # Apply coordinate validation (rotated system)
    _, _, validated_landing_x, validated_landing_y, validation_warnings = validate_coordinate_arrays(
        hit_x_list, hit_y_list, landing_x_list, landing_y_list, player_codes, court_rotation_90=True
    )
    # Log validation warnings
    if validation_warnings:
        unique_warnings = list(set(validation_warnings))
        for warning in unique_warnings:
            count = validation_warnings.count(warning)
            print(f"[VALIDATION] {count} rally end positions: {warning}")

    # Map Chinese win reasons to English (strip whitespace, print unmapped)
    win_reason_raw = rally_ends_df['win_reason'].fillna('N/A').tolist()
    win_reason_map = {
        '掛網': 'Net',
        '出界': 'Out',
        '未過網': 'Did not cross net',
        '殺球得分': 'Smash Winner',
        '挑球得分': 'Lift Winner',
        '對方失誤': 'Opponent Error',
        '發球得分': 'Service Winner',
        '接發失誤': 'Return Error',
        'N/A': 'Unknown',
        '': 'Unknown',
        '對手未過網': 'Opponent did not cross net',
        '對手出界': 'Opponent out',
        '落地致勝': 'Winner (shuttle landed)',
        '對手掛網': 'Opponent net',
    }
    unmapped = set()
    def map_reason(r):
        r_clean = r.strip() if isinstance(r, str) else r
        if r_clean in win_reason_map:
            return win_reason_map[r_clean]
        if isinstance(r_clean, str) and r_clean.isascii():
            return r_clean
        if r_clean not in win_reason_map:
            unmapped.add(r_clean)
        return 'Other'
    win_reason_english = [map_reason(r) for r in win_reason_raw]
    if unmapped:
        print(f" Unmapped win reasons: {list(unmapped)[:10]}")

    return {
        'landing_x': [float(x) for x in validated_landing_x],
        'landing_y': [float(y) for y in validated_landing_y],
        'rally_length': rally_ends_df['ball_round'].tolist(),
        'win_reason': win_reason_english,
        'set_number': rally_ends_df['set_number'].tolist()
    }

def get_player_rankings():
    """
    Creates player rankings using PageRank algorithm based on match outcomes.
    
    Returns:
        dict: Dictionary containing player rankings and network statistics
    """
    import networkx as nx
    
    df = load_data()
    
    # Create player game nodes
    games = create_player_nodes(df)
    
    # Create DataFrame for network analysis
    df_player_games = pd.DataFrame(games, columns=['Loser', 'Winner'])
    
    # Create directed graph
    G_players = nx.from_pandas_edgelist(
        df=df_player_games,
        source='Loser',
        target='Winner',
        create_using=nx.DiGraph
    )
    
    # Calculate PageRank
    player_ranks = nx.pagerank(G_players)
    
    # Sort players by rank in descending order
    sorted_player_ranks = sorted(player_ranks.items(), key=lambda item: item[1], reverse=True)
    
    # Create ranking data for display
    rankings = []
    for i, (player, rank) in enumerate(sorted_player_ranks, 1):
        rankings.append({
            'rank': i,
            'player': player,
            'pagerank_score': round(rank, 5),
            'matches_won': len(df[df['winner'] == player]),
            'matches_lost': len(df[df['loser'] == player]),
            'total_matches': len(df[df['winner'] == player]) + len(df[df['loser'] == player])
        })
    
    # Network statistics
    network_stats = {
        'total_players': G_players.number_of_nodes(),
        'total_matches': G_players.number_of_edges(),
        'avg_degree': sum(dict(G_players.degree()).values()) / G_players.number_of_nodes() if G_players.number_of_nodes() > 0 else 0
    }
    
    return {
        'rankings': rankings,
        'network_stats': network_stats
    }

def create_player_nodes(dataframe):
    """
    Reads input dataframe of badminton matches, aggregates by match ID,
    and creates a list of match outcomes (winner and loser).

    INPUT: dataframe - pandas DataFrame with badminton match data

    OUTPUT: games - array of games in format games[i] = [loser, winner]
    """
    df = dataframe.copy()
    games = []

    # Group by match ID
    grouped_matches = df.groupby('id')

    for match_id, match_df in grouped_matches:
        # Assuming 'winner' and 'loser' are consistent for all rows within a match ID
        # Get the winner and loser for the match (take the first value as they should be the same)
        winner = match_df['winner'].iloc[0]
        loser = match_df['loser'].iloc[0]

        # Append the match outcome as [loser, winner]
        games.append([loser, winner])

    return games

def get_player_network_visualization():
    """
    Creates a network visualization of players based on match outcomes.
    
    Returns:
        str: Base64 encoded image of the network visualization
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import io
    import base64
    
    df = load_data()
    
    # Create player game nodes
    games = create_player_nodes(df)
    
    # Create DataFrame for network analysis
    df_player_games = pd.DataFrame(games, columns=['Loser', 'Winner'])
    
    # Create directed graph
    G_players = nx.from_pandas_edgelist(
        df=df_player_games,
        source='Loser',
        target='Winner',
        create_using=nx.DiGraph
    )
    
    # Create visualization
    plt.figure(figsize=(12, 12), dpi=100)
    pos = nx.spring_layout(G_players, k=0.5, iterations=50)
    
    # Draw the network
    nx.draw(G_players, pos, 
            with_labels=True, 
            node_size=500, 
            font_size=8, 
            edge_color='gray', 
            alpha=0.6,
            arrows=True,
            arrowsize=10)
    
    plt.title("Badminton Player Network based on Match Outcomes")
    plt.tight_layout()
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_str

def get_statistical_radar_comparison(player_a=None, player_b=None, tournament=None, round_name=None):
    """
    Creates statistical radar comparison data for two players.
    
    Args:
        player_a (str): First player name
        player_b (str): Second player name  
        tournament (str): Tournament filter (optional)
        round_name (str): Round filter (optional)
    
    Returns:
        dict: Radar chart data with player statistics
    """
    print(f" Radar comparison requested for {player_a} vs {player_b}")
    print(f" Tournament filter: {tournament}")
    print(f" Round filter: {round_name}")
    
    df = load_data()
    print(f" Total dataset size: {len(df)}")
    
    # Filter data if tournament and round are specified
    if tournament:
        df = df[df['tournament'] == tournament]
        print(f" After tournament filter: {len(df)}")
    if round_name:
        df = df[df['round'] == round_name]
        print(f" After round filter: {len(df)}")
    
    # Get all matches for both players
    player_a_matches = df[(df['winner'] == player_a) | (df['loser'] == player_a)]
    player_b_matches = df[(df['winner'] == player_b) | (df['loser'] == player_b)]
    
    print(f" Player A matches: {len(player_a_matches)}")
    print(f" Player B matches: {len(player_b_matches)}")
    
    # Calculate statistics for Player A
    player_a_stats = calculate_player_statistics(player_a_matches, player_a)
    
    # Calculate statistics for Player B
    player_b_stats = calculate_player_statistics(player_b_matches, player_b)
    
    # Normalize values to 0-100 scale for radar chart
    max_values = {
        'matches_played': max(player_a_stats['matches_played'], player_b_stats['matches_played']),
        'wins': max(player_a_stats['wins'], player_b_stats['wins']),
        'losses': max(player_a_stats['losses'], player_b_stats['losses']),
        'points_won': max(player_a_stats['points_won'], player_b_stats['points_won']),
        'points_lost': max(player_a_stats['points_lost'], player_b_stats['points_lost']),
        'win_rate': 100  # Win rate is already a percentage
    }
    
    print(f" Max values for normalization: {max_values}")
    
    # Normalize player A stats
    player_a_normalized = {}
    for key, value in player_a_stats.items():
        if key == 'win_rate':
            player_a_normalized[key] = value
        else:
            player_a_normalized[key] = (value / max_values[key]) * 100 if max_values[key] > 0 else 0
    
    # Normalize player B stats
    player_b_normalized = {}
    for key, value in player_b_stats.items():
        if key == 'win_rate':
            player_b_normalized[key] = value
        else:
            player_b_normalized[key] = (value / max_values[key]) * 100 if max_values[key] > 0 else 0
    
    result = {
        'labels': ['Matches Played', 'Wins', 'Losses', 'Points Won', 'Points Lost', 'Win Rate %'],
        'player_a': {
            'name': player_a,
            'values': [
                player_a_normalized['matches_played'],
                player_a_normalized['wins'],
                player_a_normalized['losses'],
                player_a_normalized['points_won'],
                player_a_normalized['points_lost'],
                player_a_normalized['win_rate']
            ],
            'raw_values': player_a_stats
        },
        'player_b': {
            'name': player_b,
            'values': [
                player_b_normalized['matches_played'],
                player_b_normalized['wins'],
                player_b_normalized['losses'],
                player_b_normalized['points_won'],
                player_b_normalized['points_lost'],
                player_b_normalized['win_rate']
            ],
            'raw_values': player_b_stats
        }
    }
    
    print(f" Final result: {result}")
    return result

def calculate_player_statistics(matches_df, player_name):
    # Get unique matches (by id) to avoid counting rallies multiple times
    unique_matches = matches_df.drop_duplicates(subset=['id'])

    # Calculate match statistics
    matches_played = len(unique_matches)
    wins = len(unique_matches[unique_matches['winner'] == player_name])
    losses = len(unique_matches[unique_matches['loser'] == player_name])
    win_rate = (wins / matches_played * 100) if matches_played > 0 else 0

    # Calculate points won/lost using getpoint_player column or fallback to match scores
    points_won = 0
    points_lost = 0
    if 'getpoint_player' in matches_df.columns:
        for _, row in matches_df.iterrows():
            if row['getpoint_player'] == 'A':
                point_winner = row['winner']
            elif row['getpoint_player'] == 'B':
                point_winner = row['loser']
            else:
                continue
            if point_winner == player_name:
                points_won += 1
            else:
                points_lost += 1
    else:
        # Fallback: use final match scores (winner gets higher score, loser gets lower score)
        for _, match in unique_matches.iterrows():
            if match['winner'] == player_name:
                if 'roundscore_A' in match and 'roundscore_B' in match:
                    points_won += max(match['roundscore_A'], match['roundscore_B'])
                    points_lost += min(match['roundscore_A'], match['roundscore_B'])
            else:
                if 'roundscore_A' in match and 'roundscore_B' in match:
                    points_lost += max(match['roundscore_A'], match['roundscore_B'])
                    points_won += min(match['roundscore_A'], match['roundscore_B'])

    stats = {
        'matches_played': matches_played,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'points_won': points_won,
        'points_lost': points_lost
    }
    return stats

def get_overall_player_comparison(target_player=None):
    """
    Creates overall performance comparison for one player against all other players.
    
    Args:
        target_player (str): The player to compare against all others
    
    Returns:
        dict: Overall comparison data with rankings and statistics
    """
    df = load_data()
    
    # Get all unique players
    all_players = sorted(set(df['winner'].dropna()) | set(df['loser'].dropna()))
    
    if target_player not in all_players:
        return {'error': f'Player {target_player} not found in dataset'}
    
    # Calculate statistics for all players
    all_player_stats = {}
    for player in all_players:
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        all_player_stats[player] = calculate_player_statistics(player_matches, player)
    
    # Get target player stats
    target_stats = all_player_stats[target_player]
    
    # Calculate rankings for each metric
    rankings = {
        'matches_played': sorted(all_players, key=lambda p: all_player_stats[p]['matches_played'], reverse=True),
        'wins': sorted(all_players, key=lambda p: all_player_stats[p]['wins'], reverse=True),
        'win_rate': sorted(all_players, key=lambda p: all_player_stats[p]['win_rate'], reverse=True),
        'points_won': sorted(all_players, key=lambda p: all_player_stats[p]['points_won'], reverse=True),
        'points_lost': sorted(all_players, key=lambda p: all_player_stats[p]['points_lost'], reverse=True)
    }
    
    # Get target player's rank in each category
    target_rankings = {}
    for metric, ranked_players in rankings.items():
        try:
            rank = ranked_players.index(target_player) + 1
            total_players = len(ranked_players)
            target_rankings[metric] = {
                'rank': rank,
                'total_players': total_players,
                'percentile': round((total_players - rank + 1) / total_players * 100, 1)
            }
        except ValueError:
            target_rankings[metric] = {'rank': 0, 'total_players': 0, 'percentile': 0}
    
    # Calculate average stats across all players for comparison
    avg_stats = {}
    for metric in ['matches_played', 'wins', 'losses', 'win_rate', 'points_won', 'points_lost']:
        values = [stats[metric] for stats in all_player_stats.values()]
        avg_stats[metric] = sum(values) / len(values) if values else 0
    
    # Find top 5 players in each category
    top_players = {}
    for metric in ['matches_played', 'wins', 'win_rate', 'points_won']:
        top_5 = rankings[metric][:5]
        top_players[metric] = [
            {
                'name': player,
                'value': all_player_stats[player][metric],
                'rank': i + 1
            }
            for i, player in enumerate(top_5)
        ]
    
    return {
        'target_player': target_player,
        'target_stats': target_stats,
        'target_rankings': target_rankings,
        'average_stats': avg_stats,
        'top_players': top_players,
        'total_players': len(all_players),
        'all_players': all_players
    }

def get_shot_type_profile(player=None):
    """
    Returns the count of each shot type (smash, drop, drive, clear, net shot) for the given player.
    """
    df = load_data()
    if player:
        # Player can be 'A' or 'B' or actual name, so match both
        # Try to match by player name in winner/loser columns
        player_rows = (df['winner'] == player) | (df['loser'] == player) | (df['player'] == player)
        df = df[player_rows]
    # Standardize shot type names
    shot_type_map = {
        'smash': 'Smash',
        'drop': 'Drop',
        'drive': 'Drive',
        'clear': 'Clear',
        'net': 'Net Shot',
        'net shot': 'Net Shot',
        'netshot': 'Net Shot',
        'net drop': 'Net Shot',
        'lob': 'Clear',
        'lift': 'Clear',
        'push': 'Drive',
        'block': 'Drive',
        'kill': 'Smash',
        'defensive clear': 'Clear',
        'defensive lob': 'Clear',
        'defensive drive': 'Drive',
        'defensive drop': 'Drop',
        'offensive clear': 'Clear',
        'offensive lob': 'Clear',
        'offensive drive': 'Drive',
        'offensive drop': 'Drop',
        'offensive net': 'Net Shot',
        'defensive net': 'Net Shot',
    }
    # Lowercase and map shot types
    shot_types = df['type'].dropna().str.lower().map(lambda t: shot_type_map.get(t.strip(), t.strip()))
    # Only keep the main categories
    main_types = ['Smash', 'Drop', 'Drive', 'Clear', 'Net Shot']
    counts = {stype: 0 for stype in main_types}
    for stype in shot_types:
        if stype in counts:
            counts[stype] += 1
    return counts

def get_serve_vs_rally_efficiency(player=None):
    """
    Calculate serve vs rally point efficiency for a player.
    Analyzes performance on serve points vs rally points.
    Serve points: ball_round == 1 (immediate serve point)
    Rally points: ball_round > 1 (longer rallies)
    """
    df = load_data()
    
    if not player:
        return {
            'serve_points_won': 0,
            'serve_points_total': 0,
            'rally_points_won': 0,
            'rally_points_total': 0,
            'serve_efficiency': 0,
            'rally_efficiency': 0
        }
    
    # Filter data for the specific player
    # We need to map player name to 'A' or 'B' for each match
    player_data = []
    
    for _, match_group in df.groupby(['winner', 'loser']):
        winner = match_group['winner'].iloc[0]
        loser = match_group['loser'].iloc[0]
        
        # Check if our target player is in this match
        if player not in [winner, loser]:
            continue
            
        # Map player to 'A' or 'B'
        player_code = 'A' if player == winner else 'B'
        
        # Filter for this player's data
        match_player_data = match_group[match_group['player'] == player_code].copy()
        player_data.append(match_player_data)
    
    if not player_data:
        return {
            'serve_points_won': 0,
            'serve_points_total': 0,
            'rally_points_won': 0,
            'rally_points_total': 0,
            'serve_efficiency': 0,
            'rally_efficiency': 0
        }
    
    # Combine all player data
    player_df = pd.concat(player_data, ignore_index=True)
    
    # Filter for points (where getpoint_player is not null)
    points_data = player_df[player_df['getpoint_player'].notna()]
    
    if points_data.empty:
        return {
            'serve_points_won': 0,
            'serve_points_total': 0,
            'rally_points_won': 0,
            'rally_points_total': 0,
            'serve_efficiency': 0,
            'rally_efficiency': 0
        }
    
    # Identify serve points vs rally points based on ball_round
    serve_points = []
    rally_points = []
    
    for _, row in points_data.iterrows():
        # Check if this point was won by our player
        point_winner = row['getpoint_player']
        is_point_won = (point_winner == row['player'])
        
        # Determine if this is a serve point or rally point based on ball_round
        # ball_round == 1: Serve point (immediate serve)
        # ball_round > 1: Rally point (longer rally)
        ball_round = row.get('ball_round', 1)  # Default to 1 if not available
        
        if ball_round == 1:
            serve_points.append(is_point_won)
        else:
            rally_points.append(is_point_won)
    
    # Calculate statistics
    serve_points_won = sum(serve_points)
    serve_points_total = len(serve_points)
    rally_points_won = sum(rally_points)
    rally_points_total = len(rally_points)
    
    serve_efficiency = (serve_points_won / serve_points_total * 100) if serve_points_total > 0 else 0
    rally_efficiency = (rally_points_won / rally_points_total * 100) if rally_points_total > 0 else 0
    
    return {
        'serve_points_won': serve_points_won,
        'serve_points_total': serve_points_total,
        'rally_points_won': rally_points_won,
        'rally_points_total': rally_points_total,
        'serve_efficiency': round(serve_efficiency, 1),
        'rally_efficiency': round(rally_efficiency, 1)
    }

# --- ADVANCED KPI FUNCTIONS ---
def compute_win_rate_by_rally_length(data, player_name, player_a, player_b):
    subset = data.dropna(subset=["rally", "getpoint_player", "ball_round"])
    rally_winner = subset.groupby("rally")["getpoint_player"].last()
    rally_length = subset.groupby("rally")["ball_round"].max()
    df_rally = pd.concat([rally_winner, rally_length], axis=1).dropna()

    player_map = {"A": player_a, "B": player_b}
    df_rally["getpoint_player"] = df_rally["getpoint_player"].map(player_map)
    df_rally["won"] = df_rally["getpoint_player"] == player_name

    bins = [0, 3, 5, 8, 100]
    labels = ["1–3", "4–5", "6–8", "9+"]
    df_rally["length_bin"] = pd.cut(df_rally["ball_round"], bins=bins, labels=labels)

    result = df_rally.groupby("length_bin", observed=False)["won"].agg(["count", "sum"])
    result["win_rate_percent"] = (result["sum"] / result["count"]) * 100
    return result["win_rate_percent"].round(1).fillna(0).to_dict()

def serve_receive_performance(data, target_player, player_a, player_b):
    try:
        df_valid = data.dropna(subset=["rally", "player", "getpoint_player", "ball_round"]).copy()
        player_map = {"A": player_a, "B": player_b}
        df_valid["getpoint_player"] = df_valid["getpoint_player"].map(player_map)
        df_valid["player_full"] = df_valid["player"].map(player_map)

        first_in_rally = (
            df_valid.sort_values(["rally", "ball_round"])
            .groupby("rally")
            .first()
            .reset_index()
        )

        rally_outcome = (
            df_valid.groupby("rally")["getpoint_player"]
            .last()
            .reset_index(name="winner")
        )

        rally_summary = pd.merge(
            first_in_rally[["rally", "player_full"]], rally_outcome, on="rally"
        )
        rally_summary.rename(columns={"player_full": "server_name"}, inplace=True)
        rally_summary["is_server"] = rally_summary["server_name"] == target_player
        rally_summary["won"] = rally_summary["winner"] == target_player

        serve_win_pct = (
            rally_summary[rally_summary["is_server"]]["won"].mean() * 100
            if not rally_summary[rally_summary["is_server"]].empty else 0.0
        )
        receive_win_pct = (
            rally_summary[~rally_summary["is_server"]]["won"].mean() * 100
            if not rally_summary[~rally_summary["is_server"]].empty else 0.0
        )

        return {
            "Serve Win %": round(serve_win_pct, 1),
            "Receive Win %": round(receive_win_pct, 1),
        }

    except Exception as e:
        print(f"[Serve/Receive KPI ERROR] for {target_player}: {e}")
        return {"Serve Win %": 0.0, "Receive Win %": 0.0}

def shot_effectiveness(data, target_player, player_a, player_b):
    df = data.dropna(subset=["rally", "getpoint_player", "player", "type", "ball_round"]).copy()
    player_map = {"A": player_a, "B": player_b}
    df["getpoint_player"] = df["getpoint_player"].map(player_map)
    df["player_full"] = df["player"].map(player_map)
    df_player = df[df["player_full"] == target_player]
    last_shots = df_player.sort_values(["rally", "ball_round"]).groupby("rally").tail(1)
    rally_winners = df.groupby("rally")["getpoint_player"].last().reset_index(name="rally_winner")
    last_shots = pd.merge(last_shots, rally_winners, on="rally", how="inner")
    effectiveness = (
        last_shots.groupby("type")["rally_winner"]
        .apply(lambda x: round((x == target_player).mean() * 100, 1))
        .to_dict()
    )
    return effectiveness

def shot_selection_frequency(data, target_player, player_a, player_b):
    df_player = data[data["player"] == ("A" if target_player == player_a else "B")].copy()
    total_shots = len(df_player)
    if total_shots == 0:
        return {}
    shot_counts = df_player["type"].value_counts()
    freq_pct = (shot_counts / total_shots) * 100
    
    # Translate Chinese shot types to English
    shot_type_map = {
        '未知球種': 'Unknown Shot Type',
        'smash': 'Smash',
        'drop': 'Drop',
        'drive': 'Drive',
        'clear': 'Clear',
        'net': 'Net Shot',
        'net shot': 'Net Shot',
        'netshot': 'Net Shot',
        'net drop': 'Net Shot',
        'lob': 'Clear',
        'lift': 'Clear',
        'push': 'Drive',
        'block': 'Drive',
        'kill': 'Smash',
        'defensive clear': 'Clear',
        'defensive lob': 'Clear',
        'defensive drive': 'Drive',
        'defensive drop': 'Drop',
        'offensive clear': 'Clear',
        'offensive lob': 'Clear',
        'offensive drive': 'Drive',
        'offensive drop': 'Drop',
        'offensive net': 'Net Shot',
        'defensive net': 'Net Shot',
        'short service': 'Short Service',
        'long service': 'Long Service',
        'defensive return drive': 'Defensive Return Drive',
        'defensive return lob': 'Defensive Return Lob',
        'return net': 'Return Net'
    }
    
    def translate_shot_type(shot_type):
        return shot_type_map.get(shot_type, shot_type)
    
    translated_freq_pct = {translate_shot_type(k): v for k, v in freq_pct.round(1).to_dict().items()}
    return translated_freq_pct

def error_type_breakdown(data, target_player, player_a, player_b):
    df = data.dropna(subset=["rally", "getpoint_player", "player", "lose_reason"]).copy()
    player_map = {"A": player_a, "B": player_b}
    df["getpoint_player"] = df["getpoint_player"].map(player_map)
    df["player_full"] = df["player"].map(player_map)
    last_shots = df.sort_values(["rally", "ball_round"]).groupby("rally").tail(1)
    error_rallies = last_shots[last_shots["getpoint_player"] != target_player]
    player_errors = error_rallies[error_rallies["player_full"] == target_player]
    error_counts = player_errors["lose_reason"].value_counts()
    total = error_counts.sum()
    error_percent = (error_counts / total * 100).round(1).to_dict()
    
    # Translate Chinese error types to English
    error_type_map = {
        '掛網': 'Net',
        '出界': 'Out',
        '未過網': 'Did not cross net',
        '落點判斷失誤': 'Landing misjudgment',
        'error': 'Error',
        'net': 'Net',
        'out': 'Out'
    }
    
    def translate_error_type(error_type):
        return error_type_map.get(error_type, error_type)
    
    translated_error_percent = {translate_error_type(k): v for k, v in error_percent.items()}
    return translated_error_percent

def calculate_streaks_by_set(data, target_player, player_a, player_b):
    player_map = {"A": player_a, "B": player_b}
    data = data.dropna(subset=["rally", "getpoint_player", "set_number"])
    data["getpoint_player"] = data["getpoint_player"].map(player_map)
    streaks = {}
    for set_num in sorted(data["set_number"].unique()):
        set_data = data[data["set_number"] == set_num]
        rally_order = set_data.groupby("rally")["getpoint_player"].last().tolist()
        max_win, max_loss = 0, 0
        current_win, current_loss = 0, 0
        for winner in rally_order:
            if winner == target_player:
                current_win += 1
                current_loss = 0
            else:
                current_loss += 1
                current_win = 0
            max_win = max(max_win, current_win)
            max_loss = max(max_loss, current_loss)
        streaks[f"Set {set_num}"] = {
            "Longest Win Streak": max_win,
            "Longest Loss Streak": max_loss
        }
    return streaks

def get_score_progression(data, player_a, player_b):
    score_data = []
    data = data.sort_values(["set_number", "rally", "ball_round"])
    rallies = data.drop_duplicates(subset=["set_number", "rally"])
    for _, row in rallies.iterrows():
        set_no = row["set_number"]
        rally_no = row["rally"]
        score_a = row["roundscore_A"]
        score_b = row["roundscore_B"]
        score_data.append({
            "set": f"Set {int(set_no)}",
            "rally": int(rally_no),
            player_a: int(score_a),
            player_b: int(score_b)
        })
    df_score = pd.DataFrame(score_data)
    return df_score

def calculate_clutch_performance(data, target_player, player_a, player_b):
    data = data.dropna(subset=["rally", "getpoint_player", "roundscore_A", "roundscore_B"])
    final_shots = data.sort_values(["rally", "ball_round"]).groupby("rally").last().reset_index()
    player_map = {"A": player_a, "B": player_b}
    final_shots["getpoint_player"] = final_shots["getpoint_player"].map(player_map)
    clutch_rallies = final_shots[
        ((final_shots["roundscore_A"].between(18, 20)) |
        (final_shots["roundscore_B"].between(18, 20))) |
        ((final_shots["roundscore_A"] >= 20) & (final_shots["roundscore_B"] >= 20))
    ].copy()
    clutch_rallies["won"] = clutch_rallies["getpoint_player"] == target_player
    if len(clutch_rallies) == 0:
        return 0.0
    clutch_win_pct = round(clutch_rallies["won"].mean() * 100, 1)
    return clutch_win_pct

def compute_mean_winning_shots(df, player_name, player_a, player_b):
    """
    Compute mean number of shots in winning rallies for a player.
    Updated to use correct column names from the dataset.
    """
    try:
        # Map player names to A/B codes
        player_map = {player_a: "A", player_b: "B"}
        player_code = player_map.get(player_name, "A")
        
        # Get rallies won by this player
        df_valid = df.dropna(subset=["rally", "getpoint_player"])
        won_rallies = df_valid[df_valid['getpoint_player'] == player_code]['rally'].unique()
        
        if len(won_rallies) == 0:
            return 0.0
        
        # Count shots in those winning rallies
        winning_shots_df = df_valid[df_valid['rally'].isin(won_rallies)]
        shots_per_rally = winning_shots_df.groupby('rally').size()
        mean_winning_shots = shots_per_rally.mean()
        
        return round(mean_winning_shots, 2)
    except Exception as e:
        print(f"[Mean Winning Shots ERROR] for {player_name}: {e}")
        return 0.0

def get_rally_decision_support(df_badminton, player_a, player_b):
    """
    Aggregates KPIs and logic for the Rally Analysis Decision Support section.
    Returns a dict with:
      - executive_summary: dict of one-line insights
      - tactical_recommendations: list of dicts (KPI, recommendation)
      - expected_utility: dict (simple win probability simulation)
      - final_decision_suggestions: list of dicts (suggestion, color)
    """
    # 1. Executive Summary KPIs
    rally_stats = calculate_rally_statistics(df_badminton, player_a, player_b)
    clutch_a = calculate_clutch_performance(df_badminton, player_a, player_a, player_b)
    clutch_b = calculate_clutch_performance(df_badminton, player_b, player_a, player_b)
    streaks_a = calculate_streaks_by_set(df_badminton, player_a, player_a, player_b)
    streaks_b = calculate_streaks_by_set(df_badminton, player_b, player_a, player_b)
    radar = get_statistical_radar_comparison(player_a, player_b)
    # PageRank (get_player_rankings returns all, filter for these players)
    rankings = get_player_rankings()['rankings']
    pagerank_a = next((r['pagerank_score'] for r in rankings if r['player'] == player_a), None)
    pagerank_b = next((r['pagerank_score'] for r in rankings if r['player'] == player_b), None)
    # Error breakdown
    error_a = error_type_breakdown(df_badminton, player_a, player_a, player_b)
    error_b = error_type_breakdown(df_badminton, player_b, player_a, player_b)
    # Shot placement: use get_rally_end_zone_data if available
    try:
        rally_end_a = get_rally_end_zone_data(winner=player_a)
        rally_end_b = get_rally_end_zone_data(winner=player_b)
    except Exception:
        rally_end_a = rally_end_b = None
    # Shot type frequency (top 5)
    shot_freq_a = shot_selection_frequency(df_badminton, player_a, player_a, player_b)
    shot_freq_b = shot_selection_frequency(df_badminton, player_b, player_a, player_b)
    top5_shots_a = sorted(shot_freq_a.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_shots_b = sorted(shot_freq_b.items(), key=lambda x: x[1], reverse=True)[:5]
    # Max error type
    max_error_type_a = max(error_a.items(), key=lambda x: x[1]) if error_a else (None, None)
    max_error_type_b = max(error_b.items(), key=lambda x: x[1]) if error_b else (None, None)

    executive_summary = {
        'win_rate_a': rally_stats['player_a_win_rate'],
        'win_rate_b': rally_stats['player_b_win_rate'],
        'clutch_a': clutch_a,
        'clutch_b': clutch_b,
        'pagerank_a': pagerank_a,
        'pagerank_b': pagerank_b,
        'streaks_a': streaks_a,
        'streaks_b': streaks_b,
        'shot_placement_a': rally_end_a,
        'shot_placement_b': rally_end_b,
        'max_error_type_a': max_error_type_a,
        'max_error_type_b': max_error_type_b,
        'max_shot_types_a': top5_shots_a,
        'max_shot_types_b': top5_shots_b,
        'error_breakdown_a': error_a,
        'error_breakdown_b': error_b
    }

    # 2. Tactical Recommendations Table
    tactical_recommendations = []
    for player, clutch, pagerank, error in [
        (player_a, clutch_a, pagerank_a, error_a),
        (player_b, clutch_b, pagerank_b, error_b)
    ]:
        if clutch < 50:
            tactical_recommendations.append({'player': player, 'KPI': 'Clutch', 'recommendation': 'Improve end-game focus'})
        if pagerank and pagerank < 0.05:
            tactical_recommendations.append({'player': player, 'KPI': 'PageRank', 'recommendation': 'Increase match wins against top players'})
        if error and max(error.values(), default=0) > 30:
            tactical_recommendations.append({'player': player, 'KPI': 'Error', 'recommendation': 'Reduce unforced errors'})

    # 4. Expected Utility (simple simulation)
    # Use win rates as proxy for utility
    expected_utility = {
        player_a: rally_stats['player_a_win_rate'],
        player_b: rally_stats['player_b_win_rate']
    }

    # 5. Final Decision Suggestions (use new recommendation logic)
    final_decision_suggestions = []
    for player, clutch, pagerank, streaks, win_rate in [
        (player_a, clutch_a, pagerank_a, streaks_a, rally_stats['player_a_win_rate']),
        (player_b, clutch_b, pagerank_b, streaks_b, rally_stats['player_b_win_rate'])
    ]:
        # Map clutch score to label
        if clutch >= 60:
            clutch_label = "High"
        elif clutch >= 40:
            clutch_label = "Moderate"
        else:
            clutch_label = "Low"
        # Extract longest win streak
        if streaks and isinstance(streaks, dict):
            longest_streak = max([v.get('Longest Win Streak', 0) for v in streaks.values() if isinstance(v, dict)], default=0)
        else:
            longest_streak = 0
        # win_rate is already a fraction (0-1)
        rec = final_player_recommendation(player, pagerank if pagerank is not None else 0, clutch_label, longest_streak, win_rate/100 if win_rate > 1 else win_rate)
        final_decision_suggestions.append(rec)

    # Build tactical_recommendations_insights for the new tab
    def get_shot_landing_insight(player, top_shot):
        if top_shot and top_shot[0]:
            if 'net' in top_shot[0].lower():
                return f"Wins most points with net shots; forces opponent forward."
            elif 'smash' in top_shot[0].lower():
                return f"Wins most points with smashes; aggressive attacking style."
            elif 'clear' in top_shot[0].lower():
                return f"Wins most points with clears; pushes opponent to back court."
            else:
                return f"Wins most points with {top_shot[0]}."
        return "No dominant shot type."

    def get_error_zone_insight(max_error):
        if max_error and max_error[0]:
            if 'net' in max_error[0].lower():
                return "Net errors frequent; vulnerable in front court exchanges."
            elif 'smash' in max_error[0].lower():
                return "High unforced errors on smashes; struggles under fast rallies."
            elif 'clear' in max_error[0].lower():
                return "Errors on clears; needs to improve backcourt consistency."
            else:
                return f"Frequent errors: {max_error[0]}."
        return "No dominant error type."

    def get_clutch_insight(clutch):
        if clutch >= 60:
            return "Performs above average in clutch rallies; frequently wins at 18-20 scorelines."
        elif clutch >= 40:
            return "Clutch performance drops in tight game moments."
        else:
            return "Struggles in clutch rallies; needs end-game focus."

    def get_streak_insight(streaks):
        if not streaks:
            return "No streak data."
        win_streaks = [v['Longest Win Streak'] for v in streaks.values() if 'Longest Win Streak' in v]
        if win_streaks and max(win_streaks) >= 3:
            return "Maintains long winning streaks in at least one set."
        else:
            return "Inconsistent rally streaks; momentum is easily broken."

    def get_pagerank_insight(pagerank):
        if pagerank is None:
            return "No PageRank data."
        elif pagerank >= 0.07:
            return "High PageRank; consistent wins vs top players."
        elif pagerank >= 0.04:
            return "Mid-level PageRank due to few matches vs top-ranked opponents despite high win rate."
        else:
            return "Low PageRank reflects inconsistent performance and limited wins vs top players."

    def get_recommendation(player, clutch, pagerank, max_error, top_shot):
        recs = []
        if clutch < 50:
            recs.append("Improve end-game focus")
        if pagerank and pagerank < 0.05:
            recs.append("Increase match wins against top players")
        if max_error and max_error[1] and max_error[1] > 30:
            recs.append("Reduce unforced errors")
        if top_shot and 'net' in top_shot[0].lower():
            recs.append("Introduce variation in shot placement")
        return ", ".join(recs) if recs else "Maintain current strategy"

    tactical_recommendations_insights = []
    for player, clutch, pagerank, max_error, top_shot, streaks in [
        (player_a, clutch_a, pagerank_a, max_error_type_a, top5_shots_a[0] if top5_shots_a else (None, None), streaks_a),
        (player_b, clutch_b, pagerank_b, max_error_type_b, top5_shots_b[0] if top5_shots_b else (None, None), streaks_b)
    ]:
        tactical_recommendations_insights.append({
            'player': player,
            'max_shot_type': f"{top_shot[0]} ({top_shot[1]:.1f}%)" if top_shot and top_shot[0] else "-",
            'max_error_type': f"{max_error[0]} ({max_error[1]:.1f}%)" if max_error and max_error[0] else "-",
            'shot_landing_insight': get_shot_landing_insight(player, top_shot),
            'error_zone_insight': get_error_zone_insight(max_error),
            'clutch_insight': get_clutch_insight(clutch),
            'streak_insight': get_streak_insight(streaks),
            'pagerank_insight': get_pagerank_insight(pagerank),
            'recommendation': get_recommendation(player, clutch, pagerank, max_error, top_shot)
        })

    # Refactored expected utility calculation per provided code
    def expected_utility(df, player, player_a, player_b):
        print('Calculating EU for:', player)
        print('Unique winners:', df['winner'].unique()[:5])
        print('Unique losers:', df['loser'].unique()[:5])
        relevant_df = df[(df['winner'] == player) | (df['loser'] == player)]
        print('Relevant rows:', len(relevant_df))
        total = len(relevant_df)
        if total == 0:
            return {'Expected Utility': 0, 'P_win': 0, 'P_error': 0, 'P_loss': 0}
        win = len(relevant_df[relevant_df['winner'] == player])
        error = len(relevant_df[(relevant_df['loser'] == player) & (relevant_df['lose_reason'].notnull())])
        loss = len(relevant_df[(relevant_df['loser'] == player) & (relevant_df['lose_reason'].isnull())])
        p_win = win / total
        p_error = error / total
        p_loss = loss / total
        expected_util = p_win * 1 + p_error * -1 + p_loss * 0
        return {
            'Expected Utility': round(expected_util, 2),
            'P_win': round(p_win, 2),
            'P_error': round(p_error, 2),
            'P_loss': round(p_loss, 2)
        }

    expected_utility_a = expected_utility(df_badminton, player_a, player_a, player_b)
    expected_utility_b = expected_utility(df_badminton, player_b, player_a, player_b)

    expected_utility_dict = {
        player_a: expected_utility_a,
        player_b: expected_utility_b
    }

    return {
        'executive_summary': executive_summary,
        'tactical_recommendations': tactical_recommendations,
        'tactical_recommendations_insights': tactical_recommendations_insights,
        'expected_utility': expected_utility_dict,
        'final_decision_suggestions': final_decision_suggestions
    }

    # Add the missing expected_utility_all_matches function
    def expected_utility_all_matches(full_df, player):
        # Define shot type weights for expected utility calculation (higher = more valuable)
        shot_weights = {
            'smash': 1.2,      # Most aggressive, highest reward
            'net': 0.8,        # Defensive, lower reward
            'net shot': 0.8,   # Defensive, lower reward
            'netshot': 0.8,    # Defensive, lower reward
            'clear': 1.0,      # Neutral baseline
            'drop': 0.9,       # Tactical, moderate reward
            'drive': 1.1,      # Aggressive, high reward
        }
        # Only consider shots where this player is the hitter
        player_shots = full_df[full_df['player'] == player]
        print(f" Calculating EU for: {player}")
        print(f" Unique values in 'player' column: {full_df['player'].unique()}")
        print(f" Rows where {player} is hitter: {len(player_shots)}")
        print(f" Rows where {player} is hitter and winner: {len(player_shots[player_shots['winner'] == player])}")
        if player_shots.empty:
            return {'Expected Utility': 0, 'P_win': 0}
        shot_types = player_shots['type'].dropna().unique()
        weighted_sum = 0
        total_weight = 0
        for shot in shot_types:
            shot_df = player_shots[player_shots['type'] == shot]
            total = len(shot_df)
            if total == 0:
                continue
            # Win if this player is the winner of the rally
            win = len(shot_df[shot_df['winner'] == player])
            print(f" Shot type: {shot}, total: {total}, wins: {win}")
            p_win = win / total
            weight = shot_weights.get(shot.lower(), 1.0)
            weighted_sum += p_win * weight
            total_weight += weight
        expected_util = weighted_sum / total_weight if total_weight > 0 else 0
        # Also show overall win probability for all rallies where this player is the hitter
        total_rallies = len(player_shots)
        total_wins = len(player_shots[player_shots['winner'] == player])
        p_win_overall = total_wins / total_rallies if total_rallies > 0 else 0
        return {
            'Expected Utility': round(expected_util, 2),
            'P_win': round(p_win_overall, 2)
        }

# Add the helper function for player recommendation
def final_player_recommendation(player_name, pagerank_score, clutch_score, longest_streak, win_rate):
    """
    Generate a recommendation label and reasoning for a player's selection status.
    """
    if pagerank_score > 0.15 and clutch_score == "High" and longest_streak >= 3 and win_rate > 0.50:
        label = "✅ Green Light – Core Starter"
        reason = (
            f"{player_name} shows consistent dominance with a strong influence (PageRank {pagerank_score:.2f}), "
            f"clutch reliability, and a winning streak of {longest_streak}. Excellent win rate ({win_rate:.0%})."
        )
    elif pagerank_score > 0.07 and clutch_score in ["Moderate", "High"] and longest_streak >= 1 and win_rate > 0.30:
        label = "🟡 Yellow Light – Rotational Player"
        reason = (
            f"{player_name} maintains moderate influence (PageRank {pagerank_score:.2f}) and "
            f"can perform under pressure. Longest streak: {longest_streak}. "
            f"Solid choice for rotation (Win Rate {win_rate:.0%})."
        )
    else:
        label = "🔴 Red Light – Development Focus"
        reason = (
            f"{player_name} needs improvement across key areas. Lower influence (PageRank {pagerank_score:.2f}), "
            f"clutch performance '{clutch_score}', and shorter rally streaks (max {longest_streak}) indicate "
            f"development potential. Current win rate: {win_rate:.0%}."
        )
    return {
        'Player': player_name,
        'Label': label,
        'Reason': reason
    }
