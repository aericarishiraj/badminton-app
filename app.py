import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(REPO_ROOT, "requirements.txt")

def _ensure_requirements_installed():
    """Auto-detect Python version and install compatible packages"""
    import platform
    
    py_version = sys.version_info
    is_mac = platform.system() == "Darwin"
    is_arm64 = platform.machine() == "arm64"
    
    print(f"[Startup] Python {py_version.major}.{py_version.minor}.{py_version.micro} on {platform.system()} {platform.machine()}")
    
    if py_version.major == 3 and py_version.minor >= 13:
        packages = [
            "numpy>=2.0.0",
            "pandas>=2.2.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "scipy>=1.12.0",
            "networkx>=3.2.0",
            "opencv-python>=4.9.0",
            "Flask>=3.0.0"
        ]
        print("[Startup] Python 3.13+ detected - installing NumPy 2.x compatible packages")
    elif py_version.major == 3 and py_version.minor >= 11:
        packages = [
            "numpy==1.26.4",
            "pandas==2.2.3",
            "matplotlib==3.7.2",
            "seaborn==0.12.2",
            "scipy==1.11.4",
            "networkx==3.1",
            "opencv-python==4.8.0.76",
            "Flask==2.3.3"
        ]
        print("[Startup] Python 3.11-3.12 detected - installing stable packages")
    else:
        packages = [
            "numpy==1.24.3",
            "pandas==1.5.3",
            "matplotlib==3.6.3",
            "seaborn==0.12.2",
            "scipy==1.10.1",
            "networkx==3.0",
            "opencv-python==4.7.0.72",
            "Flask==2.3.3"
        ]
        print("[Startup] Python 3.8-3.10 detected - installing legacy compatible packages")
    
    try:
        print("[Startup] Installing compatible dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("[Startup] Dependencies installed successfully!")
    except Exception as install_error:
        print(f"[Startup] Auto-install failed: {install_error}")
        print("[Startup] Falling back to requirements.txt if available...")
        if os.path.exists(REQUIREMENTS_FILE):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
                print("[Startup] Fallback install from requirements.txt successful!")
            except Exception as fallback_error:
                print(f"[Startup] Fallback also failed: {fallback_error}")
                print("[Startup] Please install dependencies manually or contact support")
        else:
            print("[Startup] No requirements.txt found - please install dependencies manually")

_ensure_requirements_installed()

print(f"[Startup] Dependency check complete. Using Python: {sys.executable}")
if os.environ.get("AUTO_PIP_INSTALL", "0").lower() in ("1", "true", "yes"): 
    if os.path.exists(REQUIREMENTS_FILE):
        print(f"[Startup] AUTO_PIP_INSTALL enabled; installing from {REQUIREMENTS_FILE} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
            print("[Startup] Forced requirements install finished.")
        except Exception as e:
            print(f"[Startup] Forced install failed: {e}")

from flask import Flask, render_template, request, jsonify
from controllers.dashboard_controller import (
    get_dashboard_data, 
    get_all_players, 
    get_all_tournaments, 
    get_opponents_for,
    get_tournaments_for_match,
    get_rounds_for_match,
    get_shot_landing_heatmap,
    get_serve_return_placement,
    get_available_serve_types,
    get_error_zone_data,
    get_available_error_types,
    get_movement_density_heatmap,
    get_shot_type_spatial_signature,
    get_available_shot_types,
    get_rally_end_zone_data,
    get_player_rankings,
    get_player_network_visualization,
    get_statistical_radar_comparison,
    get_overall_player_comparison,
    get_shot_type_profile,
    get_serve_vs_rally_efficiency,
    compute_win_rate_by_rally_length,
    serve_receive_performance,
    shot_effectiveness,
    shot_selection_frequency,
    error_type_breakdown,
    calculate_streaks_by_set,
    calculate_clutch_performance,
    compute_mean_winning_shots,
    get_rally_decision_support,
    load_data,
    calculate_rally_statistics
)
from models.data_model import load_data
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    df = load_data()
    print(df.columns)
    print(df['lose_reason'].unique())
    print(df[df['tournament'] == 'Indonesia Masters 2020'][['winner', 'loser', 'lose_reason', 'player']].head(20))
    print(df[df['lose_reason'].notna()][['lose_reason', 'player']].head(20))
    players = get_all_players()
    tournaments = get_all_tournaments()
    return render_template('index.html', players=players, tournaments=tournaments)

@app.route('/get_opponents')
def get_opponents():
    selected_player = request.args.get('player')
    if selected_player:
        opponents = get_opponents_for(selected_player)
    else:
        opponents = []
    return jsonify(opponents)

@app.route('/get_tournaments')
def get_tournaments():
    player = request.args.get('player')
    opponent = request.args.get('opponent')
    if player and opponent:
        tournaments = get_tournaments_for_match(player, opponent)
    else:
        tournaments = []
    return jsonify(tournaments)

@app.route('/get_rounds')
def get_rounds():
    player = request.args.get('player')
    opponent = request.args.get('opponent')
    tournament = request.args.get('tournament')
    if player and opponent and tournament:
        rounds = get_rounds_for_match(player, opponent, tournament)
    else:
        rounds = []
    return jsonify(rounds)

@app.route('/dashboard', methods=['POST'])
def dashboard():
    form_data = request.form
    context = get_dashboard_data(form_data)
    
    heatmap_data = get_shot_landing_heatmap(
        player=form_data.get('player'),
        opponent=form_data.get('opponent'),
        tournament=form_data.get('tournament'),
        round_name=form_data.get('round')
    )
    context['heatmap_data'] = heatmap_data
    
    return render_template('dashboard.html', **context)

@app.route('/dashboard/kpi', methods=['POST'])
def kpi_dashboard():
    form_data = request.form
    context = get_dashboard_data(form_data)
    return render_template('kpi_dashboard.html', **context)

@app.route('/shot_heatmap')
def shot_heatmap():
    player = request.args.get('player')
    opponent = request.args.get('opponent')
    tournament = request.args.get('tournament')
    round_name = request.args.get('round')
    set_number = request.args.get('set_number')
    heatmap_data = get_shot_landing_heatmap(
        player=player,
        opponent=opponent,
        tournament=tournament,
        round_name=round_name,
        set_number=set_number
    )
    return jsonify(heatmap_data)

@app.route('/serve_return_placement')
def serve_return_placement():
    player = request.args.get('player')
    serve_type = request.args.get('serve_type')
    winner = request.args.get('winner')
    loser = request.args.get('loser')
    data = get_serve_return_placement(player=player, serve_type=serve_type, winner=winner, loser=loser)
    return jsonify(data)

@app.route('/get_available_serve_types')
def available_serve_types():
    player = request.args.get('player')
    winner = request.args.get('winner')
    loser = request.args.get('loser')
    serve_types = get_available_serve_types(winner=winner, loser=loser, player=player)
    return jsonify(serve_types)

@app.route('/error_zone_map')
def error_zone_map():
    winner = request.args.get('winner')
    loser = request.args.get('loser')
    player = request.args.get('player')
    error_type = request.args.get('error_type')
    set_number = request.args.get('set_number')

    error_type_reverse_map = {
        'Net': '掛網',
        'Out': '出界',
        'Did not cross net': '未過網',
        'Landing misjudgment': '落點判斷失誤',
        'Error': 'error',
        'Not Over Net': '未過網'
    }
    if error_type in error_type_reverse_map:
        error_type = error_type_reverse_map[error_type]

    data = get_error_zone_data(winner=winner, loser=loser, player=player, error_type=error_type, set_number=set_number)
    error_label_map = {
        'error': 'Error',
        'out': 'Out',
        'net': 'Net',
        '掛網': 'Net',
        '出界': 'Out',
        '未過網': 'Not Over Net',
        '落點判斷失誤': 'Landing misjudgment'
    }
    error_counts_display = {error_label_map.get(k, k): v for k, v in data['error_counts'].items()}
    data['error_counts'] = error_counts_display
    return jsonify(data)

@app.route('/get_available_error_types')
def available_error_types():
    winner = request.args.get('winner')
    loser = request.args.get('loser')
    player = request.args.get('player')
    tournament = request.args.get('tournament')
    round_name = request.args.get('round')
    set_number = request.args.get('set_number')
    error_types = get_available_error_types(winner=winner, loser=loser, player=player, tournament=tournament, round_name=round_name, set_number=set_number)
    error_label_map = {
        'error': 'Error',
        'out': 'Out',
        'net': 'Net',
        '掛網': 'Net',
        '出界': 'Out',
        '未過網': 'Not Over Net'
    }
    error_types_display = [{'value': et, 'label': error_label_map.get(et, et)} for et in error_types]
    return jsonify(error_types_display)

@app.route('/movement_density_heatmap')
def movement_density_heatmap():
    winner = request.args.get('winner')
    loser = request.args.get('loser')
    player = request.args.get('player')
    set_number = request.args.get('set_number')
    data = get_movement_density_heatmap(winner=winner, loser=loser, player=player, set_number=set_number)
    return jsonify(data)

@app.route('/shot_type_spatial_signature')
def shot_type_spatial_signature():
    winner = request.args.get('winner')
    loser = request.args.get('loser')
    player = request.args.get('player')
    shot_type = request.args.get('shot_type')
    set_number = request.args.get('set_number')
    data = get_shot_type_spatial_signature(winner=winner, loser=loser, player=player, shot_type=shot_type, set_number=set_number)
    return jsonify(data)

@app.route('/get_available_shot_types')
def available_shot_types():
    winner = request.args.get('winner')
    loser = request.args.get('loser')
    player = request.args.get('player')
    set_number = request.args.get('set_number')
    shot_types = get_available_shot_types(winner=winner, loser=loser, player=player, set_number=set_number)
    return jsonify(shot_types)

@app.route('/rally_end_zone')
def rally_end_zone():
    winner = request.args.get('winner')
    loser = request.args.get('loser')
    set_number = request.args.get('set_number')
    data = get_rally_end_zone_data(winner=winner, loser=loser, set_number=set_number)
    return jsonify(data)

@app.route('/player_rankings')
def player_rankings():
    data = get_player_rankings()
    return jsonify(data)

@app.route('/player_network')
def player_network():
    network_image = get_player_network_visualization()
    return jsonify({'network_image': network_image})

@app.route('/rankings')
def rankings_page():
    return render_template('rankings.html')

@app.route('/dashboard/position')
def position_dashboard():
    return render_template('position_dashboard.html')

@app.route('/statistical_radar_comparison')
def statistical_radar_comparison():
    player_a = request.args.get('player_a')
    player_b = request.args.get('player_b')
    tournament = request.args.get('tournament')
    round_name = request.args.get('round')
    
    if not player_a or not player_b:
        return jsonify({'error': 'Both player_a and player_b are required'}), 400
    
    data = get_statistical_radar_comparison(
        player_a=player_a,
        player_b=player_b,
        tournament=tournament,
        round_name=round_name
    )
    return jsonify(data)

@app.route('/overall_player_comparison')
def overall_player_comparison():
    target_player = request.args.get('player')
    
    if not target_player:
        return jsonify({'error': 'Player parameter is required'}), 400
    
    data = get_overall_player_comparison(target_player=target_player)
    return jsonify(data)

@app.route('/get_all_players')
def get_all_players_route():
    players = get_all_players()
    return jsonify(players)

@app.route('/get_all_tournaments')
def get_all_tournaments_route():
    tournaments = get_all_tournaments()
    return jsonify(tournaments)

@app.route('/debug_test')
def debug_test():
    return render_template('debug_test.html')

@app.route('/shot_type_profile')
def shot_type_profile():
    player = request.args.get('player')
    data = get_shot_type_profile(player=player)
    return jsonify(data)

@app.route('/serve_vs_rally_efficiency')
def serve_vs_rally_efficiency():
    player = request.args.get('player')
    data = get_serve_vs_rally_efficiency(player=player)
    return jsonify(data)

@app.route('/dashboard/rally', methods=['POST'])
def rally_dashboard():
    form_data = request.form
    context = get_dashboard_data(form_data)
    
    df_badminton = context['df_badminton']
    player_a = context['player_a']
    player_b = context['player_b']
    
    records = df_badminton.head(10).to_dict('records')
    context['records'] = records
    
    winrate_a = compute_win_rate_by_rally_length(df_badminton, player_a, player_a, player_b)
    winrate_b = compute_win_rate_by_rally_length(df_badminton, player_b, player_a, player_b)
    
    serve_receive_a = serve_receive_performance(df_badminton, player_a, player_a, player_b)
    serve_receive_b = serve_receive_performance(df_badminton, player_b, player_a, player_b)
    
    shot_effectiveness_a = shot_effectiveness(df_badminton, player_a, player_a, player_b)
    shot_effectiveness_b = shot_effectiveness(df_badminton, player_b, player_a, player_b)
    
    shot_freq_a = shot_selection_frequency(df_badminton, player_a, player_a, player_b)
    shot_freq_b = shot_selection_frequency(df_badminton, player_b, player_a, player_b)
    
    error_breakdown_a = error_type_breakdown(df_badminton, player_a, player_a, player_b)
    error_breakdown_b = error_type_breakdown(df_badminton, player_b, player_a, player_b)
    
    streaks_a = calculate_streaks_by_set(df_badminton, player_a, player_a, player_b)
    streaks_b = calculate_streaks_by_set(df_badminton, player_b, player_a, player_b)
    
    clutch_win_a = calculate_clutch_performance(df_badminton, player_a, player_a, player_b)
    clutch_win_b = calculate_clutch_performance(df_badminton, player_b, player_a, player_b)
    
    mean_winning_shots_a = compute_mean_winning_shots(df_badminton, player_a, player_a, player_b)
    mean_winning_shots_b = compute_mean_winning_shots(df_badminton, player_b, player_a, player_b)
    
    context['rally_decision_support'] = get_rally_decision_support(df_badminton, player_a, player_b)
    
    context.update({
        'winrate_a': winrate_a,
        'winrate_b': winrate_b,
        'serve_receive_a': serve_receive_a,
        'serve_receive_b': serve_receive_b,
        'shot_effectiveness_a': shot_effectiveness_a,
        'shot_effectiveness_b': shot_effectiveness_b,
        'shot_freq_a': shot_freq_a,
        'shot_freq_b': shot_freq_b,
        'error_breakdown_a': error_breakdown_a,
        'error_breakdown_b': error_breakdown_b,
        'streaks_a': streaks_a,
        'streaks_b': streaks_b,
        'clutch_win_a': clutch_win_a,
        'clutch_win_b': clutch_win_b,
        'mean_winning_shots_a': mean_winning_shots_a,
        'mean_winning_shots_b': mean_winning_shots_b,
    })
    
    return render_template('rally_dashboard.html', **context)

@app.route('/dashboard/shot', methods=['POST'])
def shot_dashboard():
    form_data = request.form
    context = get_dashboard_data(form_data)
    
    df_badminton = context['df_badminton']
    records = df_badminton.head(10).to_dict('records')
    context['records'] = records
    
    return render_template('shot_dashboard.html', **context)

@app.route('/get_match_stats')
def get_match_stats():
    player = request.args.get('player')
    opponent = request.args.get('opponent')
    tournament = request.args.get('tournament')
    round_name = request.args.get('round')
    
    df = load_data()
    if player and opponent:
        df = df[((df['winner'] == player) & (df['loser'] == opponent)) |
                ((df['winner'] == opponent) & (df['loser'] == player))]
    if tournament:
        df = df[df['tournament'] == tournament]
    if round_name:
        df = df[df['round'] == round_name]
    
    if df.empty:
        return jsonify({
            'avg_rallies_per_set': 0,
            'player_a': player or '',
            'player_b': opponent or '',
            'player_a_win_rate': 0,
            'player_b_win_rate': 0
        })
    
    stats = calculate_rally_statistics(df, player, opponent)
    return jsonify({
        'avg_rallies_per_set': stats['avg_rallies_per_set'],
        'player_a': player,
        'player_b': opponent,
        'player_a_win_rate': stats['player_a_win_rate'],
        'player_b_win_rate': stats['player_b_win_rate']
    })

def overlay_model_court(image_path, homography_path, x_lines_path, y_lines_path):
    img = cv2.imread(image_path)
    H = np.load(homography_path)
    x_lines = np.load(x_lines_path)
    y_lines = np.load(y_lines_path)

    court_points = []
    for x in x_lines:
        for y in y_lines:
            court_points.append([x, y])
    court_points = np.array(court_points, dtype='float32').reshape(-1, 1, 2)
    img_points = cv2.perspectiveTransform(court_points, H).reshape(-1, 2)

    img_overlay = img.copy()
    for x in x_lines:
        pts = np.array([[x, y_lines[0]], [x, y_lines[-1]]], dtype='float32').reshape(-1, 1, 2)
        pts_img = cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(int)
        cv2.line(img_overlay, tuple(pts_img[0]), tuple(pts_img[1]), (0,255,0), 2)
    for y in y_lines:
        pts = np.array([[x_lines[0], y], [x_lines[-1], y]], dtype='float32').reshape(-1, 1, 2)
        pts_img = cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(int)
        cv2.line(img_overlay, tuple(pts_img[0]), tuple(pts_img[1]), (0,255,0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Model Court Overlay')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    print("Starting Badminton Analysis App...")
    print("Open your web browser and go to: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
