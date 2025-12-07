"""
Prior estimation for RAPM calculations.

This module provides methods to generate prior estimates for players
based on box score statistics, which can be used to regularize RAPM
calculations toward more stable estimates.

Two main approaches are supported:
1. Box Score Prior (BPM-style): Use box score stats to estimate player value
2. Multi-Phase RAPM: Use first-pass RAPM as prior for second-pass
"""

import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class BoxScorePrior:
    """
    Generate prior estimates from box score statistics.

    This uses a simplified BPM-style approach where we estimate
    offensive and defensive value from traditional box score stats.
    """

    # Default coefficients (can be tuned)
    # These are rough approximations - you may want to fit these on historical data
    OFFENSIVE_COEFFICIENTS = {
        "pts_per_min": 0.5,
        "ast_per_min": 0.3,
        "orb_per_min": 0.2,
        "tov_per_min": -0.4,
        "fg_pct_adj": 0.15,  # Above league average
        "ft_pct_adj": 0.05,
        "3p_pct_adj": 0.1,
        "usg_rate": 0.02,
    }

    DEFENSIVE_COEFFICIENTS = {
        "stl_per_min": 0.4,
        "blk_per_min": 0.3,
        "drb_per_min": 0.15,
        "pf_per_min": -0.1,
    }

    def __init__(
        self,
        offensive_coeffs: Optional[Dict[str, float]] = None,
        defensive_coeffs: Optional[Dict[str, float]] = None,
    ):
        self.offensive_coeffs = offensive_coeffs or self.OFFENSIVE_COEFFICIENTS.copy()
        self.defensive_coeffs = defensive_coeffs or self.DEFENSIVE_COEFFICIENTS.copy()

    def calculate_from_boxscore(
        self, box_df: pd.DataFrame, min_minutes: float = 100.0
    ) -> Dict[int, Tuple[float, float]]:
        """
        Calculate prior estimates from box score data.

        Args:
            box_df: DataFrame with box score stats
                Expected columns: player_id, min, pts, ast, reb (or orb/drb),
                                  stl, blk, tov, pf, fga, fgm, fta, ftm, fg3a, fg3m
            min_minutes: Minimum minutes played to include player

        Returns:
            Dict mapping player_id -> (offensive_prior, defensive_prior)
        """
        priors = {}

        for _, row in box_df.iterrows():
            player_id = int(row.get("player_id", row.get("PLAYER_ID", 0)))
            minutes = float(row.get("min", row.get("MIN", 0)))

            if minutes < min_minutes or player_id == 0:
                continue

            # Calculate per-minute stats
            pts_per_min = float(row.get("pts", row.get("PTS", 0))) / minutes
            ast_per_min = float(row.get("ast", row.get("AST", 0))) / minutes
            stl_per_min = float(row.get("stl", row.get("STL", 0))) / minutes
            blk_per_min = float(row.get("blk", row.get("BLK", 0))) / minutes
            tov_per_min = float(row.get("tov", row.get("TOV", 0))) / minutes
            pf_per_min = float(row.get("pf", row.get("PF", 0))) / minutes

            # Rebounds
            orb = float(row.get("orb", row.get("ORB", row.get("OREB", 0))))
            drb = float(row.get("drb", row.get("DRB", row.get("DREB", 0))))
            if orb == 0 and drb == 0:
                reb = float(row.get("reb", row.get("REB", 0)))
                orb = reb * 0.3  # Rough split
                drb = reb * 0.7

            orb_per_min = orb / minutes
            drb_per_min = drb / minutes

            # Shooting percentages (relative to league average)
            fga = float(row.get("fga", row.get("FGA", 0)))
            fgm = float(row.get("fgm", row.get("FGM", 0)))
            fg_pct = fgm / fga if fga > 0 else 0.45
            fg_pct_adj = fg_pct - 0.45  # Relative to ~league average

            fta = float(row.get("fta", row.get("FTA", 0)))
            ftm = float(row.get("ftm", row.get("FTM", 0)))
            ft_pct = ftm / fta if fta > 0 else 0.75
            ft_pct_adj = ft_pct - 0.75

            fg3a = float(row.get("fg3a", row.get("FG3A", 0)))
            fg3m = float(row.get("fg3m", row.get("FG3M", 0)))
            fg3_pct = fg3m / fg3a if fg3a > 0 else 0.35
            fg3_pct_adj = fg3_pct - 0.35

            # Usage rate approximation
            usg_rate = fga / minutes * 48 if minutes > 0 else 0

            # Calculate offensive prior
            o_prior = (
                self.offensive_coeffs.get("pts_per_min", 0) * pts_per_min
                + self.offensive_coeffs.get("ast_per_min", 0) * ast_per_min
                + self.offensive_coeffs.get("orb_per_min", 0) * orb_per_min
                + self.offensive_coeffs.get("tov_per_min", 0) * tov_per_min
                + self.offensive_coeffs.get("fg_pct_adj", 0) * fg_pct_adj
                + self.offensive_coeffs.get("ft_pct_adj", 0) * ft_pct_adj
                + self.offensive_coeffs.get("3p_pct_adj", 0) * fg3_pct_adj
                + self.offensive_coeffs.get("usg_rate", 0) * usg_rate
            )

            # Calculate defensive prior
            d_prior = (
                self.defensive_coeffs.get("stl_per_min", 0) * stl_per_min
                + self.defensive_coeffs.get("blk_per_min", 0) * blk_per_min
                + self.defensive_coeffs.get("drb_per_min", 0) * drb_per_min
                + self.defensive_coeffs.get("pf_per_min", 0) * pf_per_min
            )

            # Scale to per-100 possessions (rough approximation)
            # Typical game has ~100 possessions per 48 minutes
            o_prior *= 100
            d_prior *= 100

            priors[player_id] = (o_prior, d_prior)

        logger.info(f"Generated priors for {len(priors)} players from box scores")
        return priors

    def fit_coefficients(
        self,
        box_df: pd.DataFrame,
        rapm_df: pd.DataFrame,
        min_minutes: float = 500.0,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Fit prior coefficients using existing RAPM data.

        This learns the relationship between box score stats and RAPM,
        which can then be used to generate better priors.

        Args:
            box_df: DataFrame with box score stats
            rapm_df: DataFrame with RAPM results (player_id, offensive_rapm, defensive_rapm)
            min_minutes: Minimum minutes for fitting

        Returns:
            Tuple of (offensive_coeffs, defensive_coeffs)
        """
        # Merge data
        merged = box_df.merge(
            rapm_df, left_on="player_id", right_on="player_id", how="inner"
        )

        # Filter by minutes
        merged = merged[merged["min"] >= min_minutes]

        if len(merged) < 50:
            logger.warning(f"Only {len(merged)} players for fitting, using defaults")
            return self.offensive_coeffs, self.defensive_coeffs

        # Build feature matrix for offensive
        X_off = []
        y_off = []

        for _, row in merged.iterrows():
            minutes = float(row["min"])
            features = [
                float(row.get("pts", 0)) / minutes,  # pts_per_min
                float(row.get("ast", 0)) / minutes,  # ast_per_min
                float(row.get("orb", row.get("OREB", 0))) / minutes,  # orb_per_min
                float(row.get("tov", 0)) / minutes,  # tov_per_min
                # Shooting percentages could be added here
            ]
            X_off.append(features)
            y_off.append(row["offensive_rapm"])

        X_off = np.array(X_off)
        y_off = np.array(y_off)

        # Fit offensive model
        off_model = Ridge(alpha=1.0)
        off_model.fit(X_off, y_off)

        offensive_coeffs = {
            "pts_per_min": off_model.coef_[0] * 100,
            "ast_per_min": off_model.coef_[1] * 100,
            "orb_per_min": off_model.coef_[2] * 100,
            "tov_per_min": off_model.coef_[3] * 100,
        }

        # Build feature matrix for defensive
        X_def = []
        y_def = []

        for _, row in merged.iterrows():
            minutes = float(row["min"])
            features = [
                float(row.get("stl", 0)) / minutes,
                float(row.get("blk", 0)) / minutes,
                float(row.get("drb", row.get("DREB", 0))) / minutes,
                float(row.get("pf", 0)) / minutes,
            ]
            X_def.append(features)
            y_def.append(row["defensive_rapm"])

        X_def = np.array(X_def)
        y_def = np.array(y_def)

        # Fit defensive model
        def_model = Ridge(alpha=1.0)
        def_model.fit(X_def, y_def)

        defensive_coeffs = {
            "stl_per_min": def_model.coef_[0] * 100,
            "blk_per_min": def_model.coef_[1] * 100,
            "drb_per_min": def_model.coef_[2] * 100,
            "pf_per_min": def_model.coef_[3] * 100,
        }

        logger.info("Fitted prior coefficients from RAPM data")
        return offensive_coeffs, defensive_coeffs


class MultiPhaseRAPM:
    """
    Multi-phase RAPM estimation.

    Phase 1: Calculate initial RAPM with standard regularization
    Phase 2: Use Phase 1 results as prior for more stable estimates
    """

    def __init__(
        self,
        phase1_lambda: float = 2500.0,
        phase2_lambda: float = 5000.0,
        prior_weight: float = 2.0,
        min_player_possessions: float = 100.0,
    ):
        self.phase1_lambda = phase1_lambda
        self.phase2_lambda = phase2_lambda
        self.prior_weight = prior_weight
        self.min_player_possessions = min_player_possessions

    def solve(self, stints_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run multi-phase RAPM estimation.

        Args:
            stints_df: DataFrame with stint data

        Returns:
            DataFrame with final RAPM results
        """
        from .rapm_solver import RAPMSolver, RAPMConfig

        # Phase 1: Initial RAPM
        logger.info("Phase 1: Initial RAPM estimation...")
        phase1_config = RAPMConfig(
            ridge_lambda=self.phase1_lambda,
            min_player_possessions=self.min_player_possessions,
            use_prior=False,
        )
        phase1_solver = RAPMSolver(config=phase1_config)
        phase1_result = phase1_solver.solve(stints_df)

        # Extract Phase 1 results as priors
        phase1_df = phase1_result.to_dataframe()
        prior_values = {}
        for _, row in phase1_df.iterrows():
            player_id = int(row["player_id"])
            prior_values[player_id] = (row["offensive_rapm"], row["defensive_rapm"])

        # Phase 2: RAPM with Phase 1 as prior
        logger.info("Phase 2: RAPM with Phase 1 prior...")
        phase2_config = RAPMConfig(
            ridge_lambda=self.phase2_lambda,
            min_player_possessions=self.min_player_possessions,
            use_prior=True,
            prior_weight=self.prior_weight,
            prior_values=prior_values,
        )
        phase2_solver = RAPMSolver(config=phase2_config)
        phase2_result = phase2_solver.solve(stints_df)

        final_df = phase2_result.to_dataframe()

        # Add phase 1 values for comparison
        final_df = final_df.merge(
            phase1_df[["player_id", "offensive_rapm", "defensive_rapm"]],
            on="player_id",
            suffixes=("", "_phase1"),
        )

        return final_df


class LuckAdjustedRAPM:
    """
    Luck-adjusted RAPM that accounts for shooting variance.

    This regresses 3-point and free throw shooting toward expected values
    to reduce noise from shooting luck.
    """

    def __init__(
        self,
        fg3_regression_factor: float = 0.5,  # Regress 3P% this much toward career/expected
        ft_regression_factor: float = 0.3,
    ):
        self.fg3_regression = fg3_regression_factor
        self.ft_regression = ft_regression_factor

    def adjust_stints(
        self,
        stints_df: pd.DataFrame,
        player_expected_fg3: Dict[int, float],
        player_expected_ft: Dict[int, float],
    ) -> pd.DataFrame:
        """
        Adjust stint scoring for shooting luck.

        This is a placeholder for a more sophisticated implementation
        that would need shot-level data to properly adjust.
        """
        # This would require more detailed data than we currently have
        # For now, return unadjusted stints
        logger.warning("Luck adjustment requires shot-level data - not yet implemented")
        return stints_df


def calculate_rapm_with_prior(
    stints_df: pd.DataFrame,
    box_df: pd.DataFrame,
    ridge_lambda: float = 2500.0,
    prior_weight: float = 1.5,
    min_player_possessions: float = 100.0,
) -> pd.DataFrame:
    """
    Convenience function for RAPM with box score prior.

    Args:
        stints_df: DataFrame with stint data
        box_df: DataFrame with box score statistics
        ridge_lambda: Ridge regularization parameter
        prior_weight: Weight for prior observations
        min_player_possessions: Minimum possessions

    Returns:
        DataFrame with RAPM results
    """
    from .rapm_solver import RAPMSolver, RAPMConfig

    # Generate priors from box scores
    prior_generator = BoxScorePrior()
    prior_values = prior_generator.calculate_from_boxscore(box_df)

    # Configure solver with priors
    config = RAPMConfig(
        ridge_lambda=ridge_lambda,
        min_player_possessions=min_player_possessions,
        use_prior=True,
        prior_weight=prior_weight,
        prior_values=prior_values,
    )

    solver = RAPMSolver(config=config)
    result = solver.solve(stints_df)

    return result.to_dataframe()


def calculate_multi_phase_rapm(
    stints_df: pd.DataFrame,
    phase1_lambda: float = 2500.0,
    phase2_lambda: float = 5000.0,
    prior_weight: float = 2.0,
) -> pd.DataFrame:
    """
    Convenience function for multi-phase RAPM.

    Args:
        stints_df: DataFrame with stint data
        phase1_lambda: Regularization for Phase 1
        phase2_lambda: Regularization for Phase 2
        prior_weight: Weight for Phase 1 prior in Phase 2

    Returns:
        DataFrame with multi-phase RAPM results
    """
    solver = MultiPhaseRAPM(
        phase1_lambda=phase1_lambda,
        phase2_lambda=phase2_lambda,
        prior_weight=prior_weight,
    )

    return solver.solve(stints_df)
