import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'static/combined_match_data_translated.csv'
df = pd.read_csv(csv_path)

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

x = last_shots['landing_x'].values
y = last_shots['landing_y'].values

plt.figure(figsize=(8, 4))
# Court dimensions in pixels (1340 x 610 for standard badminton court)
court_left, court_right = min(x.min(), 0), max(x.max(), 1340)
court_bottom, court_top = min(y.min(), 0), max(y.max(), 610)
plt.plot([court_left, court_right, court_right, court_left, court_left],
         [court_bottom, court_bottom, court_top, court_top, court_bottom], 'k--', lw=2)
plt.scatter(x, y, c='blue', s=60, label='Rally End')
plt.title('Raw Rally-End Landing Points\nMalaysia Masters 2020, AXELSEN vs MOMOTA, Set 1')
plt.xlabel('landing_x')
plt.ylabel('landing_y')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show() 