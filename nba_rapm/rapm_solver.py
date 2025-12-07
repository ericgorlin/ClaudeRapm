"""
RAPM (Regularized Adjusted Plus-Minus) Solver

Implements ridge regression to solve for player impact per 100 possessions.

The canonical approach:
1. Total RAPM: Regress MARGIN on player indicators (+1 for team, -1 for opponents)
   This gives the overall impact of having a player on the court.

2. O/D Split: Separate regressions on points scored and points allowed to estimate
   how much of the impact comes from offense vs defense. Note that these are
   approximations since the same players are on court for both ends.

Interpretation:
- O-RAPM: Points added per 100 possessions on offense (team scoring)
- D-RAPM: Points saved per 100 possessions on defense (positive = better defense)
- Total RAPM: Overall impact = O-RAPM + D-RAPM

This module is designed to be extensible for:
- Box score priors (Bayesian approach)
- Multi-phase estimation
- Different regularization strategies
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


@dataclass
class RAPMConfig:
    """Configuration for RAPM calculation."""

    ridge_lambda: float = 2500.0
    min_possessions: float = 0.5
    min_player_possessions: float = 100.0
    possession_weighted: bool = True
    per_possessions: float = 100.0
    use_prior: bool = False
    prior_weight: float = 1.0
    prior_values: Optional[Dict[int, Tuple[float, float]]] = None


@dataclass
class RAPMResult:
    """Results from RAPM calculation."""

    player_ids: List[int]
    offensive_rapm: np.ndarray
    defensive_rapm: np.ndarray

    config: RAPMConfig
    n_stints: int
    n_players: int
    total_possessions: float

    margin_r2: Optional[float] = None
    offensive_r2: Optional[float] = None
    defensive_r2: Optional[float] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        return pd.DataFrame({
            "player_id": self.player_ids,
            "offensive_rapm": self.offensive_rapm,
            "defensive_rapm": self.defensive_rapm,
            "total_rapm": self.offensive_rapm + self.defensive_rapm,
        })


class RAPMSolver:
    """
    Solves for RAPM coefficients using ridge regression.

    Uses the standard +1/-1 design matrix where:
    - Team players get +1
    - Opponent players get -1

    For O/D split:
    - Offensive regression: Y = points scored per 100 poss
    - Defensive regression: Y = points allowed per 100 poss (negated for sign convention)
    """

    def __init__(self, config: Optional[RAPMConfig] = None):
        self.config = config or RAPMConfig()
        self.player_id_to_idx: Dict[int, int] = {}
        self.idx_to_player_id: Dict[int, int] = {}

    def _build_player_index(self, stints_df: pd.DataFrame) -> None:
        """Build mapping from player IDs to matrix indices."""
        all_players = set()

        for _, row in stints_df.iterrows():
            home_lineup = row["home_lineup"]
            away_lineup = row["away_lineup"]

            if isinstance(home_lineup, (tuple, list, frozenset)):
                all_players.update(home_lineup)
            if isinstance(away_lineup, (tuple, list, frozenset)):
                all_players.update(away_lineup)

        all_players = {p for p in all_players if p and p > 0}
        self.player_id_to_idx = {pid: idx for idx, pid in enumerate(sorted(all_players))}
        self.idx_to_player_id = {idx: pid for pid, idx in self.player_id_to_idx.items()}

    def _filter_players_by_possessions(self, stints_df: pd.DataFrame) -> None:
        """Remove players with insufficient possessions."""
        player_possessions: Dict[int, float] = {}

        for _, row in stints_df.iterrows():
            poss = row["home_possessions"] + row["away_possessions"]
            home_lineup = row["home_lineup"]
            away_lineup = row["away_lineup"]

            if isinstance(home_lineup, (tuple, list, frozenset)):
                for pid in home_lineup:
                    if pid in self.player_id_to_idx:
                        player_possessions[pid] = player_possessions.get(pid, 0) + poss / 2

            if isinstance(away_lineup, (tuple, list, frozenset)):
                for pid in away_lineup:
                    if pid in self.player_id_to_idx:
                        player_possessions[pid] = player_possessions.get(pid, 0) + poss / 2

        valid_players = {
            pid for pid, poss in player_possessions.items()
            if poss >= self.config.min_player_possessions
        }

        self.player_id_to_idx = {pid: idx for idx, pid in enumerate(sorted(valid_players))}
        self.idx_to_player_id = {idx: pid for pid, idx in self.player_id_to_idx.items()}

        logger.info(f"Filtered to {len(valid_players)} players with >= {self.config.min_player_possessions} possessions")

    def _build_design_matrix(
        self, stints_df: pd.DataFrame
    ) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build design matrix for RAPM.

        Returns:
            Tuple of (X, y_scored, y_allowed, weights, possessions)
        """
        n_players = len(self.player_id_to_idx)

        valid_stints = stints_df[
            (stints_df["home_possessions"] + stints_df["away_possessions"]) / 2
            >= self.config.min_possessions
        ].copy()

        row_indices = []
        col_indices = []
        data = []
        y_scored_list = []
        y_allowed_list = []
        weights_list = []
        poss_list = []

        stint_idx = 0
        for _, row in valid_stints.iterrows():
            home_lineup = row["home_lineup"]
            away_lineup = row["away_lineup"]
            home_poss = row["home_possessions"]
            away_poss = row["away_possessions"]
            home_pts = row["home_points"]
            away_pts = row["away_points"]

            avg_poss = (home_poss + away_poss) / 2
            if avg_poss < self.config.min_possessions:
                continue

            # Home team observation
            if isinstance(home_lineup, (tuple, list, frozenset)):
                for pid in home_lineup:
                    if pid in self.player_id_to_idx:
                        row_indices.append(stint_idx)
                        col_indices.append(self.player_id_to_idx[pid])
                        data.append(1.0)

            if isinstance(away_lineup, (tuple, list, frozenset)):
                for pid in away_lineup:
                    if pid in self.player_id_to_idx:
                        row_indices.append(stint_idx)
                        col_indices.append(self.player_id_to_idx[pid])
                        data.append(-1.0)

            pts_scored = home_pts / avg_poss * self.config.per_possessions if avg_poss > 0 else 0
            pts_allowed = away_pts / avg_poss * self.config.per_possessions if avg_poss > 0 else 0

            y_scored_list.append(pts_scored)
            y_allowed_list.append(pts_allowed)
            weights_list.append(np.sqrt(avg_poss) if self.config.possession_weighted else 1.0)
            poss_list.append(avg_poss)
            stint_idx += 1

            # Away team observation (flipped)
            if isinstance(away_lineup, (tuple, list, frozenset)):
                for pid in away_lineup:
                    if pid in self.player_id_to_idx:
                        row_indices.append(stint_idx)
                        col_indices.append(self.player_id_to_idx[pid])
                        data.append(1.0)

            if isinstance(home_lineup, (tuple, list, frozenset)):
                for pid in home_lineup:
                    if pid in self.player_id_to_idx:
                        row_indices.append(stint_idx)
                        col_indices.append(self.player_id_to_idx[pid])
                        data.append(-1.0)

            pts_scored = away_pts / avg_poss * self.config.per_possessions if avg_poss > 0 else 0
            pts_allowed = home_pts / avg_poss * self.config.per_possessions if avg_poss > 0 else 0

            y_scored_list.append(pts_scored)
            y_allowed_list.append(pts_allowed)
            weights_list.append(np.sqrt(avg_poss) if self.config.possession_weighted else 1.0)
            poss_list.append(avg_poss)
            stint_idx += 1

        X = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(stint_idx, n_players),
            dtype=np.float64,
        )

        return (
            X,
            np.array(y_scored_list),
            np.array(y_allowed_list),
            np.array(weights_list),
            np.array(poss_list),
        )

    def _apply_prior(
        self, X: sparse.csr_matrix, y: np.ndarray, weights: np.ndarray, prior_values: np.ndarray
    ) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        """Add prior observations."""
        n_players = X.shape[1]
        prior_X = sparse.eye(n_players, dtype=np.float64, format="csr")
        prior_weights = np.full(n_players, self.config.prior_weight)

        X_combined = sparse.vstack([X, prior_X])
        y_combined = np.concatenate([y, prior_values])
        weights_combined = np.concatenate([weights, prior_weights])

        return X_combined, y_combined, weights_combined

    def _compute_r2(self, y: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted R-squared."""
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0

    def solve(self, stints_df: pd.DataFrame) -> RAPMResult:
        """Solve for RAPM coefficients."""
        logger.info(f"Starting RAPM solve with {len(stints_df)} stints")

        self._build_player_index(stints_df)
        logger.info(f"Found {len(self.player_id_to_idx)} unique players")

        self._filter_players_by_possessions(stints_df)

        if len(self.player_id_to_idx) == 0:
            raise ValueError("No players meet minimum possession requirements")

        n_players = len(self.player_id_to_idx)

        X, y_scored, y_allowed, weights, possessions = self._build_design_matrix(stints_df)
        y_margin = y_scored - y_allowed

        logger.info(f"Design matrix shape: {X.shape}")
        logger.info(f"Total possessions: {possessions.sum():.0f}")

        # Prepare priors
        off_priors = np.zeros(n_players)
        def_priors = np.zeros(n_players)
        if self.config.use_prior and self.config.prior_values:
            for pid, (o_prior, d_prior) in self.config.prior_values.items():
                if pid in self.player_id_to_idx:
                    idx = self.player_id_to_idx[pid]
                    off_priors[idx] = o_prior
                    def_priors[idx] = d_prior

        # Solve OFFENSIVE RAPM
        # Predicts points scored - higher = player helps team score
        logger.info("Solving offensive RAPM...")
        X_off_reg, y_off_reg, w_off_reg = (X, y_scored, weights)
        if self.config.use_prior and self.config.prior_values:
            X_off_reg, y_off_reg, w_off_reg = self._apply_prior(X, y_scored, weights, off_priors)

        off_model = Ridge(alpha=self.config.ridge_lambda, fit_intercept=True)
        off_model.fit(X_off_reg, y_off_reg, sample_weight=w_off_reg)
        off_coefs = off_model.coef_
        off_r2 = self._compute_r2(y_off_reg, off_model.predict(X_off_reg), w_off_reg)

        # Solve DEFENSIVE RAPM
        # Predicts points allowed - lower = player helps team defend
        # We negate the coefficients so positive = good defense
        logger.info("Solving defensive RAPM...")
        X_def_reg, y_def_reg, w_def_reg = (X, y_allowed, weights)
        if self.config.use_prior and self.config.prior_values:
            X_def_reg, y_def_reg, w_def_reg = self._apply_prior(X, y_allowed, weights, -def_priors)

        def_model = Ridge(alpha=self.config.ridge_lambda, fit_intercept=True)
        def_model.fit(X_def_reg, y_def_reg, sample_weight=w_def_reg)
        def_coefs = -def_model.coef_  # Negate so positive = better defense
        def_r2 = self._compute_r2(y_def_reg, def_model.predict(X_def_reg), w_def_reg)

        # Compute margin R² using combined O-D
        margin_pred = off_coefs - (-def_coefs)  # O - (-D) = O + D for margin
        margin_pred_full = X @ margin_pred
        margin_r2 = self._compute_r2(y_margin, margin_pred_full + (np.mean(y_margin) - np.mean(margin_pred_full)), weights)

        player_ids = [self.idx_to_player_id[i] for i in range(n_players)]

        result = RAPMResult(
            player_ids=player_ids,
            offensive_rapm=off_coefs,
            defensive_rapm=def_coefs,
            config=self.config,
            n_stints=X.shape[0],
            n_players=n_players,
            total_possessions=possessions.sum(),
            margin_r2=margin_r2,
            offensive_r2=off_r2,
            defensive_r2=def_r2,
        )

        logger.info(f"RAPM solve complete. O-R²={off_r2:.4f}, D-R²={def_r2:.4f}")

        return result


class MultiSeasonRAPM:
    """Calculate RAPM across multiple seasons."""

    def __init__(self, config: Optional[RAPMConfig] = None):
        self.config = config or RAPMConfig()
        self.solver = RAPMSolver(config=self.config)

    def solve_multi_season(
        self,
        season_stints: Dict[str, pd.DataFrame],
        season_weights: Optional[Dict[str, float]] = None,
    ) -> RAPMResult:
        all_stints = []

        for season, stints_df in season_stints.items():
            stints_copy = stints_df.copy()
            stints_copy["season"] = season

            if season_weights and season in season_weights:
                weight = season_weights[season]
                stints_copy["home_possessions"] *= weight
                stints_copy["away_possessions"] *= weight

            all_stints.append(stints_copy)

        combined_df = pd.concat(all_stints, ignore_index=True)

        logger.info(f"Combined {len(season_stints)} seasons with {len(combined_df)} total stints")

        return self.solver.solve(combined_df)


def calculate_rapm(
    stints_df: pd.DataFrame,
    ridge_lambda: float = 2500.0,
    min_player_possessions: float = 100.0,
    use_prior: bool = False,
    prior_values: Optional[Dict[int, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Convenience function to calculate RAPM from stint data."""
    config = RAPMConfig(
        ridge_lambda=ridge_lambda,
        min_player_possessions=min_player_possessions,
        use_prior=use_prior,
        prior_values=prior_values,
    )

    solver = RAPMSolver(config=config)
    result = solver.solve(stints_df)

    return result.to_dataframe()
