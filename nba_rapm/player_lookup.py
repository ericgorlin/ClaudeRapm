"""
Player lookup utilities for mapping IDs to names.
"""

import logging
from typing import Dict, Optional
from pathlib import Path

import pandas as pd
from nba_api.stats.static import players

logger = logging.getLogger(__name__)


class PlayerLookup:
    """
    Lookup player information by ID.

    Caches player data to avoid repeated API calls.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._players_df: Optional[pd.DataFrame] = None
        self._id_to_name: Dict[int, str] = {}
        self._cache_dir = cache_dir

    def _load_players(self) -> None:
        """Load all players from nba_api."""
        if self._players_df is not None:
            return

        all_players = players.get_players()
        self._players_df = pd.DataFrame(all_players)

        # Build ID to name mapping
        for player in all_players:
            self._id_to_name[player["id"]] = player["full_name"]

    def get_name(self, player_id: int) -> str:
        """Get player name by ID."""
        self._load_players()
        return self._id_to_name.get(player_id, f"Unknown ({player_id})")

    def get_names(self, player_ids: list) -> Dict[int, str]:
        """Get names for multiple player IDs."""
        self._load_players()
        return {pid: self.get_name(pid) for pid in player_ids}

    def add_names_to_df(self, df: pd.DataFrame, id_column: str = "player_id") -> pd.DataFrame:
        """Add player names to a DataFrame."""
        self._load_players()
        df = df.copy()
        df["player_name"] = df[id_column].apply(self.get_name)
        return df

    def search(self, name: str) -> pd.DataFrame:
        """Search for players by name."""
        self._load_players()
        mask = self._players_df["full_name"].str.lower().str.contains(name.lower())
        return self._players_df[mask]


# Global instance for convenience
_player_lookup: Optional[PlayerLookup] = None


def get_player_name(player_id: int) -> str:
    """Get player name by ID (uses global lookup)."""
    global _player_lookup
    if _player_lookup is None:
        _player_lookup = PlayerLookup()
    return _player_lookup.get_name(player_id)


def add_player_names(df: pd.DataFrame, id_column: str = "player_id") -> pd.DataFrame:
    """Add player names to DataFrame (uses global lookup)."""
    global _player_lookup
    if _player_lookup is None:
        _player_lookup = PlayerLookup()
    return _player_lookup.add_names_to_df(df, id_column)
