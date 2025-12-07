#!/usr/bin/env python3
"""
NBA RAPM Generator

Generates Regularized Adjusted Plus-Minus (RAPM) statistics
with offensive and defensive splits per 100 possessions.

Usage:
    # Generate RAPM for a single season
    python generate_rapm.py --season 2023-24

    # Generate RAPM for multiple seasons
    python generate_rapm.py --start-season 2020-21 --end-season 2023-24

    # Generate RAPM for all available seasons (1996-97 onwards)
    python generate_rapm.py --all-seasons

    # Include playoffs
    python generate_rapm.py --season 2023-24 --include-playoffs

    # Custom regularization
    python generate_rapm.py --season 2023-24 --ridge-lambda 5000
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nba_rapm.data_fetcher import NBADataFetcher
from nba_rapm.stint_processor import StintProcessor, process_season_stints
from nba_rapm.rapm_solver import RAPMSolver, RAPMConfig, MultiSeasonRAPM
from nba_rapm.player_lookup import add_player_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_available_seasons(start_year: int = 1996) -> List[str]:
    """Get list of available seasons from start_year to current."""
    now = datetime.now()
    if now.month >= 10:
        end_year = now.year
    else:
        end_year = now.year - 1

    seasons = []
    for year in range(start_year, end_year + 1):
        seasons.append(f"{year}-{str(year + 1)[-2:]}")
    return seasons


def generate_single_season_rapm(
    season: str,
    include_playoffs: bool = False,
    ridge_lambda: float = 2500.0,
    min_player_possessions: float = 100.0,
    output_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Generate RAPM for a single season.

    Args:
        season: Season string like "2023-24"
        include_playoffs: Whether to include playoff data
        ridge_lambda: Regularization parameter
        min_player_possessions: Minimum possessions for player inclusion
        output_dir: Directory to save results
        force_refresh: Force re-fetch of data

    Returns:
        DataFrame with RAPM results
    """
    logger.info(f"Generating RAPM for {season}")

    # Process regular season stints
    logger.info(f"Processing regular season stints for {season}...")
    regular_stints = process_season_stints(
        season, "Regular Season", force_refresh=force_refresh
    )

    all_stints = [regular_stints]

    # Optionally add playoffs
    if include_playoffs:
        logger.info(f"Processing playoff stints for {season}...")
        try:
            playoff_stints = process_season_stints(
                season, "Playoffs", force_refresh=force_refresh
            )
            if not playoff_stints.empty:
                all_stints.append(playoff_stints)
        except Exception as e:
            logger.warning(f"Could not fetch playoff data for {season}: {e}")

    # Combine stints
    combined_stints = pd.concat(all_stints, ignore_index=True)
    logger.info(f"Total stints: {len(combined_stints)}")

    if combined_stints.empty:
        logger.error(f"No stint data available for {season}")
        return pd.DataFrame()

    # Configure and run RAPM solver
    config = RAPMConfig(
        ridge_lambda=ridge_lambda,
        min_player_possessions=min_player_possessions,
    )

    solver = RAPMSolver(config=config)
    result = solver.solve(combined_stints)

    # Convert to DataFrame and add player names
    df = result.to_dataframe()
    df = add_player_names(df)
    df["season"] = season

    # Sort by total RAPM
    df = df.sort_values("total_rapm", ascending=False)

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"rapm_{season.replace('-', '_')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")

    return df


def generate_multi_season_rapm(
    seasons: List[str],
    include_playoffs: bool = False,
    ridge_lambda: float = 2500.0,
    min_player_possessions: float = 250.0,  # Higher for multi-season
    output_dir: Optional[Path] = None,
    force_refresh: bool = False,
    apply_recency_weight: bool = True,
) -> pd.DataFrame:
    """
    Generate RAPM across multiple seasons.

    Args:
        seasons: List of season strings
        include_playoffs: Whether to include playoff data
        ridge_lambda: Regularization parameter
        min_player_possessions: Minimum possessions for player inclusion
        output_dir: Directory to save results
        force_refresh: Force re-fetch of data
        apply_recency_weight: Weight recent seasons more heavily

    Returns:
        DataFrame with multi-season RAPM results
    """
    logger.info(f"Generating multi-season RAPM for {len(seasons)} seasons")

    season_stints = {}
    season_weights = {}

    for i, season in enumerate(seasons):
        logger.info(f"Processing {season} ({i+1}/{len(seasons)})")

        # Process regular season
        try:
            stints = process_season_stints(season, "Regular Season", force_refresh=force_refresh)

            if include_playoffs:
                try:
                    playoff_stints = process_season_stints(
                        season, "Playoffs", force_refresh=force_refresh
                    )
                    if not playoff_stints.empty:
                        stints = pd.concat([stints, playoff_stints], ignore_index=True)
                except Exception as e:
                    logger.warning(f"Could not fetch playoff data for {season}: {e}")

            if not stints.empty:
                season_stints[season] = stints

                # Recency weight: most recent season = 1.0, decreasing
                if apply_recency_weight:
                    weight = (i + 1) / len(seasons)
                else:
                    weight = 1.0
                season_weights[season] = weight

        except Exception as e:
            logger.error(f"Error processing {season}: {e}")
            continue

    if not season_stints:
        logger.error("No stint data available for any season")
        return pd.DataFrame()

    # Configure solver
    config = RAPMConfig(
        ridge_lambda=ridge_lambda,
        min_player_possessions=min_player_possessions,
    )

    # Run multi-season solver
    multi_solver = MultiSeasonRAPM(config=config)
    result = multi_solver.solve_multi_season(season_stints, season_weights)

    # Convert to DataFrame and add player names
    df = result.to_dataframe()
    df = add_player_names(df)
    df["seasons"] = f"{seasons[0]} to {seasons[-1]}"

    # Sort by total RAPM
    df = df.sort_values("total_rapm", ascending=False)

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"rapm_multi_{seasons[0]}_{seasons[-1]}.csv".replace("-", "_")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")

    return df


def generate_all_seasons_rapm(
    start_season: str = "1996-97",
    include_playoffs: bool = False,
    ridge_lambda: float = 2500.0,
    min_player_possessions: float = 100.0,
    output_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> dict:
    """
    Generate RAPM for all available seasons individually.

    Returns:
        Dict mapping season -> DataFrame with RAPM results
    """
    # Get all seasons
    start_year = int(start_season[:4])
    seasons = get_available_seasons(start_year)

    logger.info(f"Generating RAPM for {len(seasons)} seasons: {seasons[0]} to {seasons[-1]}")

    results = {}
    for season in seasons:
        try:
            df = generate_single_season_rapm(
                season=season,
                include_playoffs=include_playoffs,
                ridge_lambda=ridge_lambda,
                min_player_possessions=min_player_possessions,
                output_dir=output_dir,
                force_refresh=force_refresh,
            )
            if not df.empty:
                results[season] = df
                logger.info(f"Completed {season}: {len(df)} players")
        except Exception as e:
            logger.error(f"Error generating RAPM for {season}: {e}")
            continue

    # Create combined file with all seasons
    if output_dir and results:
        combined = pd.concat(results.values(), ignore_index=True)
        combined_file = Path(output_dir) / "rapm_all_seasons.csv"
        combined.to_csv(combined_file, index=False)
        logger.info(f"Saved combined results to {combined_file}")

    return results


def print_top_players(df: pd.DataFrame, n: int = 20) -> None:
    """Print top players by RAPM."""
    print("\n" + "=" * 80)
    print("TOP PLAYERS BY TOTAL RAPM (per 100 possessions)")
    print("=" * 80)

    top = df.head(n)
    print(f"{'Rank':<5} {'Player':<25} {'O-RAPM':<10} {'D-RAPM':<10} {'Total':<10}")
    print("-" * 60)

    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"{i:<5} {row['player_name']:<25} "
            f"{row['offensive_rapm']:>+7.2f}   {row['defensive_rapm']:>+7.2f}   "
            f"{row['total_rapm']:>+7.2f}"
        )

    print("\n" + "=" * 80)
    print("TOP OFFENSIVE PLAYERS")
    print("=" * 80)
    top_off = df.nlargest(n, "offensive_rapm")
    print(f"{'Rank':<5} {'Player':<25} {'O-RAPM':<10}")
    print("-" * 40)
    for i, (_, row) in enumerate(top_off.iterrows(), 1):
        print(f"{i:<5} {row['player_name']:<25} {row['offensive_rapm']:>+7.2f}")

    print("\n" + "=" * 80)
    print("TOP DEFENSIVE PLAYERS")
    print("=" * 80)
    top_def = df.nlargest(n, "defensive_rapm")
    print(f"{'Rank':<5} {'Player':<25} {'D-RAPM':<10}")
    print("-" * 40)
    for i, (_, row) in enumerate(top_def.iterrows(), 1):
        print(f"{i:<5} {row['player_name']:<25} {row['defensive_rapm']:>+7.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate NBA RAPM (Regularized Adjusted Plus-Minus) statistics"
    )

    # Season selection
    season_group = parser.add_mutually_exclusive_group()
    season_group.add_argument(
        "--season", type=str, help="Single season to analyze (e.g., '2023-24')"
    )
    season_group.add_argument(
        "--all-seasons", action="store_true", help="Analyze all available seasons"
    )

    parser.add_argument(
        "--start-season", type=str, default="1996-97",
        help="Start season for range (default: 1996-97)"
    )
    parser.add_argument(
        "--end-season", type=str, help="End season for range"
    )

    # Options
    parser.add_argument(
        "--include-playoffs", action="store_true", help="Include playoff games"
    )
    parser.add_argument(
        "--ridge-lambda", type=float, default=2500.0,
        help="Ridge regularization parameter (default: 2500)"
    )
    parser.add_argument(
        "--min-possessions", type=float, default=100.0,
        help="Minimum possessions for player inclusion (default: 100)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Output directory for results (default: output)"
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Force re-fetch of all data"
    )
    parser.add_argument(
        "--multi-season", action="store_true",
        help="Combine multiple seasons into single RAPM calculation"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top players to display (default: 20)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all_seasons:
        # Generate for all seasons
        results = generate_all_seasons_rapm(
            start_season=args.start_season,
            include_playoffs=args.include_playoffs,
            ridge_lambda=args.ridge_lambda,
            min_player_possessions=args.min_possessions,
            output_dir=output_dir,
            force_refresh=args.force_refresh,
        )

        # Print summary for most recent season
        if results:
            most_recent = list(results.keys())[-1]
            print(f"\nResults for most recent season ({most_recent}):")
            print_top_players(results[most_recent], args.top_n)

    elif args.season:
        # Single season
        df = generate_single_season_rapm(
            season=args.season,
            include_playoffs=args.include_playoffs,
            ridge_lambda=args.ridge_lambda,
            min_player_possessions=args.min_possessions,
            output_dir=output_dir,
            force_refresh=args.force_refresh,
        )

        if not df.empty:
            print_top_players(df, args.top_n)

    elif args.start_season and args.end_season:
        # Season range
        start_year = int(args.start_season[:4])
        end_year = int(args.end_season[:4])
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year + 1)]

        if args.multi_season:
            # Combine into single RAPM calculation
            df = generate_multi_season_rapm(
                seasons=seasons,
                include_playoffs=args.include_playoffs,
                ridge_lambda=args.ridge_lambda,
                min_player_possessions=args.min_possessions,
                output_dir=output_dir,
                force_refresh=args.force_refresh,
            )

            if not df.empty:
                print_top_players(df, args.top_n)
        else:
            # Individual seasons
            for season in seasons:
                df = generate_single_season_rapm(
                    season=season,
                    include_playoffs=args.include_playoffs,
                    ridge_lambda=args.ridge_lambda,
                    min_player_possessions=args.min_possessions,
                    output_dir=output_dir,
                    force_refresh=args.force_refresh,
                )

                if not df.empty:
                    print(f"\n{season}:")
                    print_top_players(df, args.top_n)

    else:
        # Default: current season
        now = datetime.now()
        if now.month >= 10:
            current_year = now.year
        else:
            current_year = now.year - 1
        current_season = f"{current_year}-{str(current_year + 1)[-2:]}"

        df = generate_single_season_rapm(
            season=current_season,
            include_playoffs=args.include_playoffs,
            ridge_lambda=args.ridge_lambda,
            min_player_possessions=args.min_possessions,
            output_dir=output_dir,
            force_refresh=args.force_refresh,
        )

        if not df.empty:
            print_top_players(df, args.top_n)


if __name__ == "__main__":
    main()
