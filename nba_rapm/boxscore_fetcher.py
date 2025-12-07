"""
Box score data fetcher for prior calculations.

Fetches season totals and per-game statistics for players.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    playercareerstats,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "boxscores"


class BoxScoreFetcher:
    """Fetches and caches box score data for RAPM prior calculations."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        request_delay: float = 0.6,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay

    def _rate_limit(self):
        """Sleep to respect rate limits."""
        time.sleep(self.request_delay)

    def get_season_totals(
        self,
        season: str,
        season_type: str = "Regular Season",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get season totals for all players.

        Args:
            season: Season string like "2023-24"
            season_type: "Regular Season" or "Playoffs"
            force_refresh: Re-fetch even if cached

        Returns:
            DataFrame with player season totals
        """
        cache_file = self.cache_dir / f"totals_{season}_{season_type.replace(' ', '_')}.parquet"

        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        logger.info(f"Fetching season totals for {season} {season_type}")
        self._rate_limit()

        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star=season_type,
                per_mode_detailed="Totals",
            )
            df = stats.get_data_frames()[0]

            # Standardize column names
            df = df.rename(columns={
                "PLAYER_ID": "player_id",
                "PLAYER_NAME": "player_name",
                "TEAM_ID": "team_id",
                "TEAM_ABBREVIATION": "team",
                "GP": "games",
                "MIN": "min",
                "PTS": "pts",
                "AST": "ast",
                "REB": "reb",
                "OREB": "orb",
                "DREB": "drb",
                "STL": "stl",
                "BLK": "blk",
                "TOV": "tov",
                "PF": "pf",
                "FGA": "fga",
                "FGM": "fgm",
                "FG_PCT": "fg_pct",
                "FG3A": "fg3a",
                "FG3M": "fg3m",
                "FG3_PCT": "fg3_pct",
                "FTA": "fta",
                "FTM": "ftm",
                "FT_PCT": "ft_pct",
            })

            df.to_parquet(cache_file)
            return df
        except Exception as e:
            logger.error(f"Error fetching season totals: {e}")
            raise

    def get_career_stats(self, player_id: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get career statistics for a player.

        Args:
            player_id: NBA player ID
            force_refresh: Re-fetch even if cached

        Returns:
            DataFrame with career stats by season
        """
        cache_file = self.cache_dir / f"career_{player_id}.parquet"

        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        self._rate_limit()

        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            df = career.get_data_frames()[0]

            df.to_parquet(cache_file)
            return df
        except Exception as e:
            logger.error(f"Error fetching career stats for {player_id}: {e}")
            raise


def get_season_boxscores(
    season: str,
    season_type: str = "Regular Season",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Convenience function to get season box score totals.

    Args:
        season: Season string
        season_type: "Regular Season" or "Playoffs"
        cache_dir: Optional cache directory

    Returns:
        DataFrame with player season totals
    """
    fetcher = BoxScoreFetcher(cache_dir=cache_dir)
    return fetcher.get_season_totals(season, season_type)
