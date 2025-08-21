from models.data_model import load_data
import pandas as pd

df = load_data()
print("Columns:", df.columns.tolist())

print(f"\nTotal unique matches (by id): {df['id'].nunique()}")
print(f"Unique winners: {df['winner'].unique()}")
print(f"Unique losers: {df['loser'].unique()}")

print(f"\nSet number distribution:")
print(df['set_number'].value_counts().sort_index())

first_match = df[df['id'] == df['id'].iloc[0]]
winner = first_match['winner'].iloc[0]
loser = first_match['loser'].iloc[0]
tournament = first_match['tournament'].iloc[0]
round_name = first_match['round'].iloc[0]

print(f"\nTesting with specific match:")
print(f"Winner: {winner}")
print(f"Loser: {loser}")
print(f"Tournament: {tournament}")
print(f"Round: {round_name}")

player_matches = (
    ((df['winner'] == winner) & (df['loser'] == loser)) |
    ((df['winner'] == loser) & (df['loser'] == winner))
)
filtered_df = df[player_matches]
print(f"After player/opponent filter: {filtered_df.shape}")

filtered_df = filtered_df[filtered_df['tournament'] == tournament]
print(f"After tournament filter: {filtered_df.shape}")

filtered_df = filtered_df[filtered_df['round'] == round_name]
print(f"After round filter: {filtered_df.shape}")

set2_filtered = filtered_df[filtered_df['set_number'] == 2]
print(f"After set 2 filter: {set2_filtered.shape}")

point_shots = set2_filtered[set2_filtered['win_reason'].notna() | set2_filtered['lose_reason'].notna()]
print(f"Point shots in this match set 2: {len(point_shots)}")

if 'id' in point_shots.columns and 'ball_round' in point_shots.columns and 'rally' in point_shots.columns:
    # Get the final shot of each rally (highest ball_round per rally)
    last_shots = point_shots.sort_values(['id', 'rally', 'ball_round']).groupby(['id', 'rally']).tail(1)
    print(f"Last shots per (id, rally) in this match set 2: {len(last_shots)}")
    
    print(f"Number of final shots: {len(last_shots)}")
    
    print(f"\nRally distribution in this match set 2:")
    rally_counts = last_shots['rally'].value_counts().sort_index()
    print(rally_counts)
    
    print("\nSample rally data:")
    print(last_shots[['id', 'rally', 'ball_round', 'win_reason', 'landing_x', 'landing_y']].head(10).to_string())
else:
    print("Required columns not found")

if len(last_shots) > 0:
    print(f"\nCoordinate ranges in this match set 2 final shots:")
    print(f"landing_x: {last_shots['landing_x'].min():.2f} to {last_shots['landing_x'].max():.2f}")
    print(f"landing_y: {last_shots['landing_y'].min():.2f} to {last_shots['landing_y'].max():.2f}")
mask = (
    (df['tournament'] == 'Malaysia Masters 2020') &
    (((df['winner'] == 'Viktor AXELSEN') & (df['loser'] == 'Kento MOMOTA')) |
     ((df['winner'] == 'Kento MOMOTA') & (df['loser'] == 'Viktor AXELSEN')))
    & (df['set_number'] == 1)
)
match_df = df[mask].copy()

match_df = match_df[match_df['rally'].notna() & match_df['ball_round'].notna()]

last_shots = match_df.sort_values(['rally', 'ball_round']).groupby('rally').tail(1)

last_shots = last_shots[(last_shots['win_reason'].notna()) | (last_shots['lose_reason'].notna())]

print(f"Total rallies in set 1: {last_shots.shape[0]}")
print(last_shots[['rally', 'ball_round', 'player', 'win_reason', 'lose_reason', 'landing_x', 'landing_y']].to_string(index=False)) 