# NBA RAPM Calculator

Calculate Regularized Adjusted Plus-Minus (RAPM) with offensive and defensive splits per 100 possessions for NBA players.

## Features

- **True RAPM Calculation**: Uses ridge regression on stint-level data with proper possession weighting
- **O-RAPM / D-RAPM Splits**: Separate offensive and defensive impact ratings
- **30 Years of Data**: Supports all seasons from 1996-97 onwards (when PBP data became available)
- **Automatic Data Caching**: Fetches and caches data from NBA Stats API
- **Extensible**: Built-in support for box score priors and multi-phase estimation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Generate RAPM for current season
python generate_rapm.py

# Generate RAPM for a specific season
python generate_rapm.py --season 2023-24

# Generate RAPM for all available seasons (1996-97 onwards)
python generate_rapm.py --all-seasons

# Include playoff games
python generate_rapm.py --season 2023-24 --include-playoffs
```

## Usage

### Command Line

```bash
# Single season
python generate_rapm.py --season 2023-24

# Season range
python generate_rapm.py --start-season 2020-21 --end-season 2023-24

# Multi-season pooled RAPM (combines data across seasons)
python generate_rapm.py --start-season 2020-21 --end-season 2023-24 --multi-season

# Custom regularization
python generate_rapm.py --season 2023-24 --ridge-lambda 5000

# Custom minimum possessions
python generate_rapm.py --season 2023-24 --min-possessions 250
```

### Python API

```python
from nba_rapm.stint_processor import process_season_stints
from nba_rapm.rapm_solver import RAPMSolver, RAPMConfig
from nba_rapm.player_lookup import add_player_names

# Get stint data for a season
stints_df = process_season_stints("2023-24", "Regular Season")

# Configure RAPM solver
config = RAPMConfig(
    ridge_lambda=2500.0,
    min_player_possessions=100.0,
)

# Solve for RAPM
solver = RAPMSolver(config=config)
result = solver.solve(stints_df)

# Get results as DataFrame
df = result.to_dataframe()
df = add_player_names(df)

# Top players by total RAPM
print(df.sort_values("total_rapm", ascending=False).head(20))
```

## RAPM Methodology

### What is RAPM?

RAPM (Regularized Adjusted Plus-Minus) measures how many points a player adds to (or subtracts from) their team's margin per 100 possessions, accounting for teammates and opponents.

### How it works

1. **Stint Extraction**: Games are broken into "stints" - periods where the lineup doesn't change
2. **Possession Calculation**: Each stint's possessions are calculated using the Dean Oliver formula
3. **Design Matrix**: A matrix X is created where each row is a stint, and each column is a player (+1 if on offense, -1 if on defense)
4. **Ridge Regression**: Separate regressions for offense and defense:
   - Y_off = points scored per 100 possessions
   - Y_def = points allowed per 100 possessions
   - Weighted by sqrt(possessions) for proper variance handling
5. **Regularization**: Ridge penalty (L2) prevents overfitting with collinear lineups

### Interpreting Results

- **O-RAPM**: Points added per 100 possessions on offense (higher = better)
- **D-RAPM**: Points saved per 100 possessions on defense (higher = better, means fewer points allowed)
- **Total RAPM**: O-RAPM - D-RAPM (since defensive contribution is "saving" points)

A league-average player has RAPM of ~0 in all categories.

## Extensions

### Box Score Prior

Add a prior based on box score statistics:

```python
from nba_rapm.priors import BoxScorePrior, calculate_rapm_with_prior
from nba_rapm.boxscore_fetcher import get_season_boxscores

# Get box score data
box_df = get_season_boxscores("2023-24")

# Calculate RAPM with box score prior
result = calculate_rapm_with_prior(
    stints_df,
    box_df,
    ridge_lambda=2500.0,
    prior_weight=1.5,
)
```

### Multi-Phase RAPM

Use initial RAPM estimates as priors for a second pass:

```python
from nba_rapm.priors import calculate_multi_phase_rapm

result = calculate_multi_phase_rapm(
    stints_df,
    phase1_lambda=2500.0,
    phase2_lambda=5000.0,
    prior_weight=2.0,
)
```

### Custom Regularization

Adjust the ridge parameter based on your needs:
- **Lower lambda (1000-2000)**: Less regularization, more variance, better for large samples
- **Higher lambda (3000-5000)**: More regularization, more stable, better for smaller samples
- **Multi-season**: Use lower lambda since you have more data

## Project Structure

```
nba_rapm/
├── __init__.py           # Package initialization
├── data_fetcher.py       # NBA API data fetching with caching
├── stint_processor.py    # Play-by-play to stint conversion
├── rapm_solver.py        # Core RAPM ridge regression
├── priors.py             # Box score priors and multi-phase RAPM
├── boxscore_fetcher.py   # Box score data for priors
└── player_lookup.py      # Player ID to name mapping

data/
└── cache/                # Cached API responses (auto-created)
    ├── games/
    ├── pbp/
    ├── boxscores/
    └── stints/

output/                   # Generated RAPM results (auto-created)
```

## Data Notes

- Play-by-play data is available from **1996-97** season onwards
- First fetch for a season may take 30-60 minutes due to API rate limits
- Subsequent runs use cached data and complete quickly
- Cache is stored in `data/cache/` and can be deleted to re-fetch

## Known Limitations

1. **Lineup tracking**: The stint processor infers lineups from play-by-play events, which may occasionally miss players who don't touch the ball
2. **Early seasons**: PBP data quality improves over time; 1996-2000 may have more noise
3. **Possession estimates**: Uses the Dean Oliver formula which is an approximation

## Contributing

The codebase is designed for extension. Key extension points:

1. **New priors**: Add to `priors.py`
2. **Alternative regularization**: Modify `RAPMSolver` in `rapm_solver.py`
3. **Data sources**: Add new fetchers following `data_fetcher.py` pattern

## License

MIT
