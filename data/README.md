# Data

The application loads five processed files from `data/processed/`:

| File | Grain | Contents |
| --- | --- | --- |
| `feature_table.csv` | one team per match | rolling form, head-to-head, rest, venue context |
| `train.csv` | one team per match | earlier seasons |
| `val.csv` | one team per match | penultimate season |
| `test.csv` | one team per match | most recent season |
| `home_matches_eda.csv` | one home team per match | retrieval-friendly match view |
| `player_game_eda.csv` | one player per game | per-game player stats + targets |

## Raw files

The large raw exports are **not committed**. Download them from the AFL data
source and place them in `data/raw/` with these exact names:

```text
data/raw/
├── afl_players_info_raw.csv
├── afl_players_round_by_round_stats_raw.csv
├── afl_players_seasonal_stats_raw.csv
└── team_matches_home_away_raw.csv
```

The player-info file is small enough that it is already included.

## Rebuilding the processed files

```bash
python scripts/prepare_data.py
```

This rebuilds `player_game_eda.csv`, the rolling match features, the
home-match retrieval table, and the chronological train/validation/test split.
All rolling features are shifted by one match so no future information leaks
into the features.
