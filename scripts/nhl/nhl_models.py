"""
NHL prop prediction models.

Shared module for training and inference.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, poisson
from sklearn.isotonic import IsotonicRegression


class SimpleSOGModel:
    """
    Simple SOG model using player season averages.

    For Phase B, uses a Normal distribution based on:
    - Season shots per game mean
    - Position-based variance

    Future enhancements:
    - Game logs for rolling averages
    - TOI splits (EV/PP)
    - Opponent defense metrics
    - Home/away splits
    """

    def __init__(self):
        self.position_std_map = {
            "C": 1.5,   # Centers: moderate variance
            "W": 1.4,   # Wings: moderate variance
            "D": 0.8,   # Defense: lower variance
            "F": 1.4,   # Forwards: moderate variance
        }
        self.default_std = 1.2
        self.skater_stats = None
        self.calibrator = None  # Isotonic regression calibrator

    def fit(self, skater_df: pd.DataFrame):
        """
        Fit model on season stats.

        For Phase B, just stores reference stats.
        Future: Train on game logs with historical data.
        """
        self.skater_stats = skater_df.set_index("player")[
            ["shots", "games_played", "position", "toi_per_game"]
        ].copy()

        # Compute shots per game
        self.skater_stats["shots_per_game"] = (
            self.skater_stats["shots"] / self.skater_stats["games_played"]
        )

        return self

    def calibrate(self, props_df: pd.DataFrame):
        """
        Calibrate model using isotonic regression.

        Args:
            props_df: DataFrame with columns [player, point, side, consensus_prob]
        """
        # Get raw model predictions
        raw_probs = self.predict_batch(props_df)

        # Get consensus probabilities (calibration targets)
        consensus_probs = props_df["consensus_prob"].values

        # Fit isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_probs, consensus_probs)

        return self

    def predict_prob_over(self, player: str, line: float, calibrated: bool = True) -> float:
        """
        Predict probability of going OVER a line.

        Uses Normal distribution: N(mean=shots_per_game, std=position_std)
        P(X > line) = 1 - CDF(line)

        Args:
            player: Player name
            line: Prop line
            calibrated: Whether to apply calibration (default True)
        """
        # Normalize player name
        player_norm = " ".join(player.strip().lower().split())

        if self.skater_stats is None or player_norm not in self.skater_stats.index:
            # Unknown player: use league average (50%)
            return 0.5

        stats = self.skater_stats.loc[player_norm]
        mean_sog = stats["shots_per_game"]
        position = stats.get("position", "F")

        # Position-based standard deviation
        std_sog = self.position_std_map.get(position, self.default_std)

        # P(X > line) using Normal CDF
        prob_over = 1 - norm.cdf(line, loc=mean_sog, scale=std_sog)

        # Apply calibration if available
        if calibrated and self.calibrator is not None:
            prob_over = self.calibrator.predict([prob_over])[0]

        return prob_over

    def predict_batch(self, props_df: pd.DataFrame, calibrated: bool = False) -> pd.Series:
        """
        Predict probabilities for a batch of props.

        Input: DataFrame with columns [player, market_std, side, point]
        Output: Series of probabilities

        Args:
            props_df: Props DataFrame
            calibrated: Whether to apply calibration (default False for training)
        """
        probs = []

        for _, row in props_df.iterrows():
            player = row["player"]
            line = row["point"]
            side = row["side"]

            if side == "Over":
                prob = self.predict_prob_over(player, line, calibrated=calibrated)
            else:  # Under
                prob = 1 - self.predict_prob_over(player, line, calibrated=calibrated)

            probs.append(prob)

        return pd.Series(probs, index=props_df.index)


class SimpleScoringModel:
    """
    Simple scoring model for Goals/Assists/Points.

    Uses Poisson distribution based on:
    - Season rate per game (goals/assists/points per game)
    - Position-based adjustments

    For 0.5 lines (common for goals), uses Bernoulli approximation.
    """

    def __init__(self, stat_name: str):
        """
        Initialize model for a specific stat.

        Args:
            stat_name: "goals", "assists", or "points"
        """
        self.stat_name = stat_name
        self.skater_stats = None
        self.calibrator = None  # Isotonic regression calibrator

    def fit(self, skater_df: pd.DataFrame):
        """Fit model on season stats."""
        stat_col_map = {
            "goals": "goals",
            "assists": "assists",
            "points": "points",
        }

        stat_col = stat_col_map.get(self.stat_name)
        if stat_col not in skater_df.columns:
            raise ValueError(f"Stat column {stat_col} not found in data")

        self.skater_stats = skater_df.set_index("player")[
            [stat_col, "games_played", "position"]
        ].copy()

        # Compute rate per game
        self.skater_stats[f"{self.stat_name}_per_game"] = (
            self.skater_stats[stat_col] / self.skater_stats["games_played"]
        )

        return self

    def calibrate(self, props_df: pd.DataFrame):
        """
        Calibrate model using isotonic regression.

        Args:
            props_df: DataFrame with columns [player, point, side, consensus_prob]
        """
        # Get raw model predictions
        raw_probs = self.predict_batch(props_df)

        # Get consensus probabilities (calibration targets)
        consensus_probs = props_df["consensus_prob"].values

        # Fit isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_probs, consensus_probs)

        return self

    def predict_prob_over(self, player: str, line: float, calibrated: bool = True) -> float:
        """
        Predict probability of going OVER a line.

        For integer lines: Uses Poisson CDF
        For 0.5 lines (goals): Uses Bernoulli approximation from rate

        Args:
            player: Player name
            line: Prop line
            calibrated: Whether to apply calibration (default True)
        """
        # Normalize player name
        player_norm = " ".join(player.strip().lower().split())

        if self.skater_stats is None or player_norm not in self.skater_stats.index:
            # Unknown player: use 50%
            return 0.5

        stats = self.skater_stats.loc[player_norm]
        rate_per_game = stats[f"{self.stat_name}_per_game"]

        # For very low lines (0.5), use Bernoulli approximation
        if line == 0.5:
            # P(X >= 1) = 1 - P(X = 0) = 1 - e^(-λ)
            prob_over = 1 - poisson.pmf(0, rate_per_game)
        else:
            # For higher lines, use Poisson survival function
            # P(X > line) = 1 - P(X <= line) = 1 - CDF(floor(line))
            prob_over = 1 - poisson.cdf(int(line), rate_per_game)

        # Apply calibration if available
        if calibrated and self.calibrator is not None:
            prob_over = self.calibrator.predict([prob_over])[0]

        return prob_over

    def predict_batch(self, props_df: pd.DataFrame, calibrated: bool = False) -> pd.Series:
        """
        Predict probabilities for a batch of props.

        Args:
            props_df: Props DataFrame
            calibrated: Whether to apply calibration (default False for training)
        """
        probs = []

        for _, row in props_df.iterrows():
            player = row["player"]
            line = row["point"]
            side = row["side"]

            if side == "Over":
                prob = self.predict_prob_over(player, line, calibrated=calibrated)
            else:  # Under
                prob = 1 - self.predict_prob_over(player, line, calibrated=calibrated)

            probs.append(prob)

        return pd.Series(probs, index=props_df.index)
