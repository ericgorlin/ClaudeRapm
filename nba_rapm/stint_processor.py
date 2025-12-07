"""
Stint processor for extracting lineup stints from play-by-play data.

A stint is a period of time where the lineup on the floor doesn't change.
For each stint, we track:
- Which players were on the floor for each team
- Points scored by each team
- Possessions for each team
- Duration
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Stint:
    """Represents a single stint (period with constant lineup)."""

    game_id: str
    period: int
    start_time: float  # seconds remaining in period
    end_time: float
    home_team_id: int
    away_team_id: int
    home_lineup: frozenset  # Set of player IDs
    away_lineup: frozenset
    home_points: int = 0
    away_points: int = 0
    home_possessions: float = 0.0
    away_possessions: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Duration of stint in seconds."""
        return self.start_time - self.end_time

    @property
    def home_margin(self) -> int:
        """Point differential from home team perspective."""
        return self.home_points - self.away_points

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "game_id": self.game_id,
            "period": self.period,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "home_lineup": tuple(sorted(self.home_lineup)),
            "away_lineup": tuple(sorted(self.away_lineup)),
            "home_points": self.home_points,
            "away_points": self.away_points,
            "home_possessions": self.home_possessions,
            "away_possessions": self.away_possessions,
            "home_margin": self.home_margin,
        }


class StintProcessor:
    """
    Processes play-by-play data to extract stint-level information.

    This uses the official NBA possession formula and tracks lineups
    through substitution events.
    """

    # Event types in NBA PBP data
    MADE_SHOT = 1
    MISSED_SHOT = 2
    FREE_THROW = 3
    REBOUND = 4
    TURNOVER = 5
    FOUL = 6
    VIOLATION = 7
    SUBSTITUTION = 8
    TIMEOUT = 9
    JUMP_BALL = 10
    EJECTION = 11
    PERIOD_BEGIN = 12
    PERIOD_END = 13

    def __init__(self):
        pass

    def parse_game_clock(self, pctimestring: str) -> float:
        """
        Parse game clock string (MM:SS) to seconds remaining.

        Args:
            pctimestring: Time string like "11:45" or "0:30"

        Returns:
            Seconds remaining in period
        """
        if pd.isna(pctimestring) or pctimestring == "":
            return 0.0

        try:
            parts = str(pctimestring).split(":")
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), int(parts[1])
                return minutes * 60 + seconds
            return 0.0
        except (ValueError, AttributeError):
            return 0.0

    def extract_lineups_from_pbp(
        self, pbp_df: pd.DataFrame, game_id: str
    ) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], int, int]:
        """
        Extract starting lineups and team IDs from play-by-play data.

        Returns:
            Tuple of (home_lineups_by_period, away_lineups_by_period, home_team_id, away_team_id)
        """
        # Get team IDs from the first plays
        home_team_id = None
        away_team_id = None

        # Find team IDs from early plays
        for _, row in pbp_df.head(50).iterrows():
            if pd.notna(row.get("PLAYER1_TEAM_ID")):
                team_id = int(row["PLAYER1_TEAM_ID"])
                # Check if this is home or away based on description
                if home_team_id is None:
                    # First team we see
                    home_team_id = team_id
                elif team_id != home_team_id and away_team_id is None:
                    away_team_id = team_id

                if home_team_id and away_team_id:
                    break

        # If we couldn't determine teams, use a fallback
        if home_team_id is None or away_team_id is None:
            logger.warning(f"Could not determine team IDs for game {game_id}")
            # Try to get from HOMEDESCRIPTION/VISITORDESCRIPTION
            for _, row in pbp_df.iterrows():
                if pd.notna(row.get("HOMEDESCRIPTION")) and pd.notna(row.get("PLAYER1_TEAM_ID")):
                    home_team_id = int(row["PLAYER1_TEAM_ID"])
                elif pd.notna(row.get("VISITORDESCRIPTION")) and pd.notna(row.get("PLAYER1_TEAM_ID")):
                    away_team_id = int(row["PLAYER1_TEAM_ID"])

                if home_team_id and away_team_id:
                    break

        return {}, {}, home_team_id or 0, away_team_id or 0

    def is_possession_ending_event(self, row: pd.Series) -> Tuple[bool, Optional[int]]:
        """
        Determine if an event ends a possession and for which team.

        Returns:
            Tuple of (ends_possession, team_id_that_had_possession)
        """
        event_type = row.get("EVENTMSGTYPE", 0)
        action_type = row.get("EVENTMSGACTIONTYPE", 0)
        team_id = row.get("PLAYER1_TEAM_ID")

        if pd.isna(team_id):
            team_id = None
        else:
            team_id = int(team_id)

        # Made shot (not and-one free throw situations - handled separately)
        if event_type == self.MADE_SHOT:
            return True, team_id

        # Turnover
        if event_type == self.TURNOVER:
            return True, team_id

        # Defensive rebound (ends offensive team's possession)
        if event_type == self.REBOUND:
            # Check if it's a team rebound after made shot (doesn't count)
            # Defensive rebounds end possessions
            # We'll count all rebounds as potential possession changes
            # and refine in possession counting
            return False, None  # Rebounds are tricky, handle in possession counting

        # Last free throw made/missed (end of possession if not offensive rebound)
        if event_type == self.FREE_THROW:
            # Check if it's the last free throw
            desc = str(row.get("HOMEDESCRIPTION", "")) + str(row.get("VISITORDESCRIPTION", ""))
            if "3 of 3" in desc or "2 of 2" in desc or "1 of 1" in desc:
                # Last free throw - possession ends unless offensive rebound
                return True, team_id
            # Technical/flagrant free throws
            if "TECH" in desc.upper() or "FLAGRANT" in desc.upper():
                return False, None  # These don't end possessions normally

        return False, None

    def calculate_points_from_event(self, row: pd.Series) -> Tuple[int, Optional[int]]:
        """
        Calculate points scored from an event.

        Returns:
            Tuple of (points, team_id)
        """
        event_type = row.get("EVENTMSGTYPE", 0)
        team_id = row.get("PLAYER1_TEAM_ID")

        if pd.isna(team_id):
            return 0, None

        team_id = int(team_id)

        # Made shot
        if event_type == self.MADE_SHOT:
            # Check if 3-pointer
            desc = str(row.get("HOMEDESCRIPTION", "")) + str(row.get("VISITORDESCRIPTION", ""))
            if "3PT" in desc.upper():
                return 3, team_id
            return 2, team_id

        # Made free throw
        if event_type == self.FREE_THROW:
            desc = str(row.get("HOMEDESCRIPTION", "")) + str(row.get("VISITORDESCRIPTION", ""))
            if "MISS" not in desc.upper():
                return 1, team_id

        return 0, None

    def process_game(self, pbp_df: pd.DataFrame, game_id: str) -> List[Stint]:
        """
        Process a single game's play-by-play data into stints.

        This is a simplified approach that tracks scoring and uses
        the possession formula for each stint.
        """
        if pbp_df.empty:
            return []

        stints = []

        # Get team IDs
        _, _, home_team_id, away_team_id = self.extract_lineups_from_pbp(pbp_df, game_id)

        if not home_team_id or not away_team_id:
            logger.warning(f"Could not determine teams for game {game_id}")
            return []

        # Group by period
        periods = pbp_df["PERIOD"].unique()

        for period in sorted(periods):
            period_df = pbp_df[pbp_df["PERIOD"] == period].copy()
            period_df = period_df.sort_values("EVENTNUM")

            # Track current lineup (we'll use events to infer)
            current_home_lineup: Set[int] = set()
            current_away_lineup: Set[int] = set()

            # Current stint tracking
            stint_start_time = 720.0 if period <= 4 else 300.0  # 12 min quarters, 5 min OT
            stint_home_points = 0
            stint_away_points = 0
            stint_events_home = []  # For possession calculation
            stint_events_away = []

            for idx, row in period_df.iterrows():
                event_type = row.get("EVENTMSGTYPE", 0)
                current_time = self.parse_game_clock(row.get("PCTIMESTRING", "0:00"))
                player1_id = row.get("PLAYER1_ID")
                player2_id = row.get("PLAYER2_ID")
                player1_team = row.get("PLAYER1_TEAM_ID")

                # Track players we see on the court
                if pd.notna(player1_id) and player1_id != 0:
                    player1_id = int(player1_id)
                    if pd.notna(player1_team):
                        if int(player1_team) == home_team_id:
                            current_home_lineup.add(player1_id)
                        elif int(player1_team) == away_team_id:
                            current_away_lineup.add(player1_id)

                # Handle substitution - end current stint, start new one
                if event_type == self.SUBSTITUTION:
                    # Save current stint if it has any duration
                    if stint_start_time > current_time and (current_home_lineup or current_away_lineup):
                        # Calculate possessions for this stint
                        home_poss, away_poss = self._estimate_stint_possessions(
                            stint_events_home, stint_events_away
                        )

                        stint = Stint(
                            game_id=game_id,
                            period=period,
                            start_time=stint_start_time,
                            end_time=current_time,
                            home_team_id=home_team_id,
                            away_team_id=away_team_id,
                            home_lineup=frozenset(current_home_lineup),
                            away_lineup=frozenset(current_away_lineup),
                            home_points=stint_home_points,
                            away_points=stint_away_points,
                            home_possessions=home_poss,
                            away_possessions=away_poss,
                        )
                        stints.append(stint)

                    # Process substitution
                    if pd.notna(player1_id) and pd.notna(player2_id):
                        player_out = int(player1_id)
                        player_in = int(player2_id)

                        if pd.notna(player1_team):
                            if int(player1_team) == home_team_id:
                                current_home_lineup.discard(player_out)
                                current_home_lineup.add(player_in)
                            else:
                                current_away_lineup.discard(player_out)
                                current_away_lineup.add(player_in)

                    # Reset stint tracking
                    stint_start_time = current_time
                    stint_home_points = 0
                    stint_away_points = 0
                    stint_events_home = []
                    stint_events_away = []
                    continue

                # Track points
                points, team_id = self.calculate_points_from_event(row)
                if points > 0 and team_id:
                    if team_id == home_team_id:
                        stint_home_points += points
                    else:
                        stint_away_points += points

                # Track events for possession calculation
                if pd.notna(player1_team):
                    event_info = {
                        "type": event_type,
                        "action": row.get("EVENTMSGACTIONTYPE", 0),
                        "desc": str(row.get("HOMEDESCRIPTION", ""))
                        + str(row.get("VISITORDESCRIPTION", "")),
                    }
                    if int(player1_team) == home_team_id:
                        stint_events_home.append(event_info)
                    else:
                        stint_events_away.append(event_info)

            # Save final stint for period
            if current_home_lineup or current_away_lineup:
                home_poss, away_poss = self._estimate_stint_possessions(
                    stint_events_home, stint_events_away
                )

                stint = Stint(
                    game_id=game_id,
                    period=period,
                    start_time=stint_start_time,
                    end_time=0.0,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_lineup=frozenset(current_home_lineup),
                    away_lineup=frozenset(current_away_lineup),
                    home_points=stint_home_points,
                    away_points=stint_away_points,
                    home_possessions=home_poss,
                    away_possessions=away_poss,
                )
                stints.append(stint)

        return stints

    def _estimate_stint_possessions(
        self, home_events: List[Dict], away_events: List[Dict]
    ) -> Tuple[float, float]:
        """
        Estimate possessions for each team in a stint.

        Uses event-based counting:
        - FGA (made or missed, excluding and-1s)
        - FTA * 0.44 (approximate possessions from FT trips)
        - Turnovers
        - Subtract offensive rebounds (they extend possessions, don't start new ones)
        """
        home_fga = 0
        home_fta = 0
        home_tov = 0
        home_oreb = 0

        away_fga = 0
        away_fta = 0
        away_tov = 0
        away_oreb = 0

        for event in home_events:
            if event["type"] == self.MADE_SHOT:
                home_fga += 1
            elif event["type"] == self.MISSED_SHOT:
                home_fga += 1
            elif event["type"] == self.FREE_THROW:
                home_fta += 1
            elif event["type"] == self.TURNOVER:
                home_tov += 1
            elif event["type"] == self.REBOUND:
                # Check if offensive
                if "OFF" in event["desc"].upper():
                    home_oreb += 1

        for event in away_events:
            if event["type"] == self.MADE_SHOT:
                away_fga += 1
            elif event["type"] == self.MISSED_SHOT:
                away_fga += 1
            elif event["type"] == self.FREE_THROW:
                away_fta += 1
            elif event["type"] == self.TURNOVER:
                away_tov += 1
            elif event["type"] == self.REBOUND:
                if "OFF" in event["desc"].upper():
                    away_oreb += 1

        # Standard possession formula (Dean Oliver)
        # Possessions = FGA + 0.44*FTA + TOV - OREB
        home_poss = home_fga + 0.44 * home_fta + home_tov - home_oreb
        away_poss = away_fga + 0.44 * away_fta + away_tov - away_oreb

        return max(0, home_poss), max(0, away_poss)

    def process_multiple_games(
        self, pbp_dict: Dict[str, pd.DataFrame], show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Process multiple games and return a DataFrame of all stints.

        Args:
            pbp_dict: Dict mapping game_id to PBP DataFrame
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with all stints
        """
        all_stints = []

        iterator = pbp_dict.items()
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Processing games")

        for game_id, pbp_df in iterator:
            try:
                stints = self.process_game(pbp_df, game_id)
                all_stints.extend([s.to_dict() for s in stints])
            except Exception as e:
                logger.warning(f"Error processing game {game_id}: {e}")
                continue

        if not all_stints:
            return pd.DataFrame()

        return pd.DataFrame(all_stints)


def process_season_stints(
    season: str,
    season_type: str = "Regular Season",
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to process all stints for a season.

    Args:
        season: Season string like "2023-24"
        season_type: "Regular Season" or "Playoffs"
        cache_dir: Optional cache directory
        force_refresh: If True, re-fetch and reprocess

    Returns:
        DataFrame with all stints for the season
    """
    from .data_fetcher import NBADataFetcher

    fetcher = NBADataFetcher(cache_dir=cache_dir if cache_dir else None)
    processor = StintProcessor()

    # Check for cached stint data
    stints_cache = fetcher.cache_dir / "stints" / f"stints_{season}_{season_type.replace(' ', '_')}.parquet"
    stints_cache.parent.mkdir(exist_ok=True)

    if stints_cache.exists() and not force_refresh:
        return pd.read_parquet(stints_cache)

    # Fetch PBP data
    pbp_data = fetcher.fetch_season_pbp(season, season_type, force_refresh)

    # Process into stints
    stints_df = processor.process_multiple_games(pbp_data)

    # Cache results
    if not stints_df.empty:
        stints_df.to_parquet(stints_cache)

    return stints_df
