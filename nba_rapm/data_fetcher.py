"""
Data fetching module for NBA play-by-play and game data.

Uses the nba_api package to fetch data from NBA Stats API.
Implements caching to avoid re-fetching data.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import (
    leaguegamelog,
    playbyplayv2,
    boxscoreadvancedv3,
    commonplayerinfo,
    leaguegamefinder,
)
from nba_api.stats.static import players, teams

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


class NBADataFetcher:
    """Fetches and caches NBA game and play-by-play data."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        request_delay: float = 0.6,  # Be nice to the API
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay

        # Create subdirectories
        (self.cache_dir / "games").mkdir(exist_ok=True)
        (self.cache_dir / "pbp").mkdir(exist_ok=True)
        (self.cache_dir / "boxscores").mkdir(exist_ok=True)

    def _rate_limit(self):
        """Sleep to respect rate limits."""
        time.sleep(self.request_delay)

    def get_season_games(
        self, season: str, season_type: str = "Regular Season", force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get all games for a season.

        Args:
            season: Season string like "2023-24"
            season_type: "Regular Season" or "Playoffs"
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with game information
        """
        cache_file = self.cache_dir / "games" / f"games_{season}_{season_type.replace(' ', '_')}.parquet"

        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        logger.info(f"Fetching games for {season} {season_type}")
        self._rate_limit()

        try:
            game_log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
            )
            df = game_log.get_data_frames()[0]

            # Save to cache
            df.to_parquet(cache_file)
            return df
        except Exception as e:
            logger.error(f"Error fetching games for {season}: {e}")
            raise

    def get_play_by_play(self, game_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get play-by-play data for a specific game.

        Args:
            game_id: NBA game ID (e.g., "0022300001")
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with play-by-play data
        """
        cache_file = self.cache_dir / "pbp" / f"pbp_{game_id}.parquet"

        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        self._rate_limit()

        try:
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            df = pbp.get_data_frames()[0]

            # Save to cache
            df.to_parquet(cache_file)
            return df
        except Exception as e:
            logger.error(f"Error fetching PBP for game {game_id}: {e}")
            raise

    def get_boxscore_advanced(self, game_id: str, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get advanced boxscore for a game (includes possessions).

        Args:
            game_id: NBA game ID
            force_refresh: If True, re-fetch even if cached

        Returns:
            Dict with 'team' and 'player' DataFrames
        """
        cache_file = self.cache_dir / "boxscores" / f"boxscore_{game_id}.parquet"

        if cache_file.exists() and not force_refresh:
            df = pd.read_parquet(cache_file)
            return {"combined": df}

        self._rate_limit()

        try:
            boxscore = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
            dfs = boxscore.get_data_frames()

            # Combine player and team stats
            if len(dfs) >= 2:
                combined = pd.concat([dfs[0], dfs[1]], ignore_index=True) if len(dfs[0]) > 0 else dfs[1]
            else:
                combined = dfs[0] if len(dfs) > 0 else pd.DataFrame()

            combined.to_parquet(cache_file)
            return {"combined": combined}
        except Exception as e:
            logger.error(f"Error fetching boxscore for game {game_id}: {e}")
            raise

    def get_all_seasons(self, start_season: str = "1996-97", end_season: Optional[str] = None) -> List[str]:
        """
        Generate list of season strings from start to end.

        Note: Play-by-play data is generally available from 1996-97 onwards.
        """
        if end_season is None:
            # Determine current season
            now = datetime.now()
            if now.month >= 10:
                end_year = now.year
            else:
                end_year = now.year - 1
            end_season = f"{end_year}-{str(end_year + 1)[-2:]}"

        seasons = []
        start_year = int(start_season[:4])
        end_year = int(end_season[:4])

        for year in range(start_year, end_year + 1):
            seasons.append(f"{year}-{str(year + 1)[-2:]}")

        return seasons

    def fetch_season_pbp(
        self,
        season: str,
        season_type: str = "Regular Season",
        force_refresh: bool = False,
        max_games: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all play-by-play data for a season.

        Args:
            season: Season string like "2023-24"
            season_type: "Regular Season" or "Playoffs"
            force_refresh: If True, re-fetch even if cached
            max_games: Limit number of games (for testing)

        Returns:
            Dict mapping game_id to PBP DataFrame
        """
        games_df = self.get_season_games(season, season_type, force_refresh)

        # Get unique game IDs
        game_ids = games_df["GAME_ID"].unique()
        if max_games:
            game_ids = game_ids[:max_games]

        pbp_data = {}
        for game_id in tqdm(game_ids, desc=f"Fetching PBP for {season}"):
            try:
                pbp_data[game_id] = self.get_play_by_play(game_id, force_refresh)
            except Exception as e:
                logger.warning(f"Failed to fetch PBP for {game_id}: {e}")
                continue

        return pbp_data

    @staticmethod
    def get_player_info(player_id: int) -> Dict[str, Any]:
        """Get player information by ID."""
        player = players.find_player_by_id(player_id)
        return player if player else {}

    @staticmethod
    def get_team_info(team_id: int) -> Dict[str, Any]:
        """Get team information by ID."""
        team = teams.find_team_by_id(team_id)
        return team if team else {}

    @staticmethod
    def get_all_players() -> List[Dict[str, Any]]:
        """Get all NBA players."""
        return players.get_players()

    @staticmethod
    def get_all_teams() -> List[Dict[str, Any]]:
        """Get all NBA teams."""
        return teams.get_teams()


def get_available_seasons() -> List[str]:
    """
    Returns list of seasons with available PBP data.
    PBP data is generally available from 1996-97 season onwards.
    """
    fetcher = NBADataFetcher()
    return fetcher.get_all_seasons(start_season="1996-97")
