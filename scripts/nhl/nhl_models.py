"""
NHL prop prediction models.

Shared module for training and inference.
"""

import sys
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

    def fit(self, skater_df: pd.DataFrame, game_logs: pd.DataFrame = None):
        """
        Fit model on season stats or game logs.

        Args:
            skater_df: Season aggregate stats (fallback for players without game logs)
            game_logs: Game-by-game logs (preferred, for rolling averages)
        """
        # If game logs provided, compute rolling averages
        if game_logs is not None and len(game_logs) > 0:
            game_log_stats = self._compute_rolling_stats(game_logs)

            # If season stats also provided, merge them as fallback
            if not skater_df.empty:
                # Prepare season stats
                season_stats = skater_df.copy()
                season_stats["player"] = season_stats["player"].apply(
                    lambda x: " ".join(x.strip().lower().split()) if isinstance(x, str) else ""
                )
                season_stats = season_stats.set_index("player")

                # Compute shots per game from season stats
                season_stats["shots_per_game"] = (
                    season_stats["shots"] / season_stats["games_played"].replace(0, 1)
                )

                # Select only players NOT in game logs
                players_in_game_logs = set(game_log_stats.index)
                players_in_season_only = season_stats.index.difference(players_in_game_logs)

                # Add season-only players to model
                season_only = season_stats.loc[players_in_season_only, ["shots_per_game", "position", "games_played"]].copy()

                # Combine: game logs (preferred) + season stats (fallback)
                self.skater_stats = pd.concat([game_log_stats, season_only])

                print(f"[Model] Using {len(game_log_stats)} players from game logs + {len(season_only)} from season stats", file=sys.stderr)
            else:
                # Only game logs, no season stats
                self.skater_stats = game_log_stats
        else:
            # Fallback: use season aggregates only
            self.skater_stats = skater_df.set_index("player")[
                ["shots", "games_played", "position", "toi_per_game"]
            ].copy()

            # Compute shots per game
            self.skater_stats["shots_per_game"] = (
                self.skater_stats["shots"] / self.skater_stats["games_played"]
            )

        return self

    def _compute_rolling_stats(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling SOG averages from game logs.

        For each player, compute:
        - L5 average (last 5 games)
        - L10 average (last 10 games)
        - Home/away splits
        - Season average (fallback)
        """
        # Sort by player and date (most recent last)
        game_logs = game_logs.sort_values(["player", "game_date"])

        player_stats = []

        for player, player_games in game_logs.groupby("player"):
            n_games = len(player_games)

            # Get position (assume constant for player)
            position = player_games["position"].iloc[0] if "position" in player_games.columns else "F"

            # Season average
            season_avg = player_games["shots"].mean() if n_games > 0 else 0.0

            # L5 average (last 5 games)
            l5_avg = player_games["shots"].tail(5).mean() if n_games >= 3 else season_avg

            # L10 average (last 10 games)
            l10_avg = player_games["shots"].tail(10).mean() if n_games >= 5 else l5_avg

            # Home/away splits
            home_games = player_games[player_games["is_home"]]
            away_games = player_games[~player_games["is_home"]]

            home_avg = home_games["shots"].mean() if len(home_games) > 0 else season_avg
            away_avg = away_games["shots"].mean() if len(away_games) > 0 else season_avg

            # Use L5 as primary rate (most recent form)
            primary_rate = l5_avg if n_games >= 3 else (l10_avg if n_games >= 2 else season_avg)

            player_stats.append({
                "player": player,
                "position": position,
                "games_played": n_games,
                "shots_per_game": primary_rate,
                "shots_l5": l5_avg,
                "shots_l10": l10_avg,
                "shots_home": home_avg,
                "shots_away": away_avg,
                "shots_season": season_avg,
            })

        df = pd.DataFrame(player_stats)
        return df.set_index("player")

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

    def fit(self, skater_df: pd.DataFrame, game_logs: pd.DataFrame = None):
        """
        Fit model on season stats or game logs.

        Args:
            skater_df: Season aggregate stats (fallback for players without game logs)
            game_logs: Game-by-game logs (preferred, for rolling averages)
        """
        stat_col_map = {
            "goals": "goals",
            "assists": "assists",
            "points": "points",
        }

        stat_col = stat_col_map.get(self.stat_name)

        # If game logs provided, compute rolling averages
        if game_logs is not None and len(game_logs) > 0:
            if stat_col not in game_logs.columns:
                raise ValueError(f"Stat column {stat_col} not found in game logs")

            game_log_stats = self._compute_rolling_stats(game_logs, stat_col)

            # If season stats also provided, merge them as fallback
            if not skater_df.empty and stat_col in skater_df.columns:
                # Prepare season stats
                season_stats = skater_df.copy()
                season_stats["player"] = season_stats["player"].apply(
                    lambda x: " ".join(x.strip().lower().split()) if isinstance(x, str) else ""
                )
                season_stats = season_stats.set_index("player")

                # Compute rate per game from season stats
                season_stats[f"{self.stat_name}_per_game"] = (
                    season_stats[stat_col] / season_stats["games_played"].replace(0, 1)
                )

                # Select only players NOT in game logs
                players_in_game_logs = set(game_log_stats.index)
                players_in_season_only = season_stats.index.difference(players_in_game_logs)

                # Add season-only players to model
                season_only = season_stats.loc[players_in_season_only, [f"{self.stat_name}_per_game", "position", "games_played"]].copy()

                # Combine: game logs (preferred) + season stats (fallback)
                self.skater_stats = pd.concat([game_log_stats, season_only])
            else:
                # Only game logs, no season stats
                self.skater_stats = game_log_stats
        else:
            # Fallback: use season aggregates only
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

    def _compute_rolling_stats(self, game_logs: pd.DataFrame, stat_col: str) -> pd.DataFrame:
        """
        Compute rolling averages from game logs.

        For each player, compute:
        - L5 average (last 5 games)
        - L10 average (last 10 games)
        - Home/away splits
        - Season average (fallback)

        Args:
            game_logs: DataFrame with columns [player, game_date, goals, assists, points, is_home]
            stat_col: Stat column to compute rolling average for

        Returns:
            DataFrame indexed by player with rolling stats
        """
        # Sort by player and date (most recent last)
        game_logs = game_logs.sort_values(["player", "game_date"])

        player_stats = []

        for player, player_games in game_logs.groupby("player"):
            n_games = len(player_games)

            # Get position (assume constant for player)
            position = player_games["position"].iloc[0] if "position" in player_games.columns else "F"

            # Season average
            season_avg = player_games[stat_col].mean() if n_games > 0 else 0.0

            # L5 average (last 5 games)
            l5_avg = player_games[stat_col].tail(5).mean() if n_games >= 3 else season_avg

            # L10 average (last 10 games)
            l10_avg = player_games[stat_col].tail(10).mean() if n_games >= 5 else l5_avg

            # Home/away splits
            home_games = player_games[player_games["is_home"]]
            away_games = player_games[~player_games["is_home"]]

            home_avg = home_games[stat_col].mean() if len(home_games) > 0 else season_avg
            away_avg = away_games[stat_col].mean() if len(away_games) > 0 else season_avg

            # Use L5 as primary rate (most recent form)
            # Fallback to L10 if insufficient L5 data, then season
            primary_rate = l5_avg if n_games >= 3 else (l10_avg if n_games >= 2 else season_avg)

            player_stats.append({
                "player": player,
                "position": position,
                "games_played": n_games,
                f"{self.stat_name}_per_game": primary_rate,
                f"{self.stat_name}_l5": l5_avg,
                f"{self.stat_name}_l10": l10_avg,
                f"{self.stat_name}_home": home_avg,
                f"{self.stat_name}_away": away_avg,
                f"{self.stat_name}_season": season_avg,
            })

        df = pd.DataFrame(player_stats)
        return df.set_index("player")

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
