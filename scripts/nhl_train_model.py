"""
Train model to predict NHL team totals
Uses team-level aggregated features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


def select_features(team_df):
    """
    Select features for modeling
    """
    feature_cols = [
        # Rolling averages
        'goals_l5', 'goals_l10', 'goals_ewm',
        'shots_l5', 'shots_l10', 'shots_ewm',
        'pp_goals_l5', 'pp_goals_l10',
        'save_pct_l5', 'save_pct_l10',

        # Recent performance
        'fwd_goals_mean', 'fwd_shots_mean', 'fwd_points_mean',
        'fwd_pp_goals_sum',

        # Defense
        'def_blocked_shots_sum', 'def_plus_minus_mean',

        # Goalie
        'goalie_save_pct_first',

        # Opponent strength
        'opp_allows_goals_against_per_game',
        'opp_allows_goalie_save_pct_first',

        # Home/away
        'is_home',
        'home_avg_goals_for',
        'away_avg_goals_for'
    ]

    # Filter to only columns that exist
    available_features = [col for col in feature_cols if col in team_df.columns]

    return available_features


def train_model(features_path, model_type='ridge', output_dir='data/nhl/models'):
    """
    Train team totals model
    """
    print(f"Loading team features from {features_path}...")
    team_df = pd.read_csv(features_path)

    # Remove rows without enough history (first few games)
    team_df = team_df.dropna(subset=['goals_l5', 'goals_l10'])

    print(f"Training on {len(team_df)} team-games")

    # Select features
    feature_cols = select_features(team_df)
    print(f"\nUsing {len(feature_cols)} features:")
    for feat in feature_cols:
        print(f"  - {feat}")

    X = team_df[feature_cols].copy()
    y = team_df['goals_for'].copy()

    # Check for missing data and warn
    missing_before = X.isna().sum()
    if missing_before.sum() > 0:
        print(f"⚠️  WARNING: Training data has {missing_before.sum()} missing values across {(missing_before > 0).sum()} columns")
        print(f"   Columns with missing data: {list(missing_before[missing_before > 0].index)}")
        print(f"   Using column means as fallback")

    # Fill any remaining NaN with column means (fallback)
    X = X.fillna(X.mean())

    # Time series split (preserve temporal order)
    tscv = TimeSeriesSplit(n_splits=5)

    # Train models
    models = {
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    model = models.get(model_type, models['ridge'])

    print(f"\nTraining {model_type} model...")

    # Cross-validation
    mae_scores = []
    rmse_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"  Fold {fold+1}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    print(f"\nCross-validation results:")
    print(f"  Average MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    print(f"  Average RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")

    # Train final model on all data
    print(f"\nTraining final model on all data...")
    model.fit(X, y)

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 most important features:")
        print(feature_importance.head(10).to_string(index=False))

    elif hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)

        print(f"\nTop 10 features by coefficient magnitude:")
        print(feature_importance.head(10).to_string(index=False))

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_type}_team_totals.pkl')
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")

    # Save feature list
    feature_list_path = os.path.join(output_dir, 'feature_list.txt')
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Saved feature list to {feature_list_path}")

    return model, feature_cols


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NHL team totals model')
    parser.add_argument('--input', default='data/nhl/processed/team_features.csv', help='Input features CSV')
    parser.add_argument('--model-type', default='ridge', choices=['ridge', 'lasso', 'elastic', 'rf', 'gbm'], help='Model type')
    parser.add_argument('--output-dir', default='data/nhl/models', help='Output directory')

    args = parser.parse_args()

    train_model(args.input, args.model_type, args.output_dir)
