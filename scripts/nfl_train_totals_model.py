"""
Train NFL totals model using team features
Similar to NHL totals but with EPA-based features
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

def train_totals_model(features_path='data/nfl/processed/team_features.csv',
                       model_type='ridge',
                       output_dir='data/nfl/models'):
    """
    Train team totals model
    """
    print("Loading team features...")
    df = pd.read_csv(features_path)

    # Filter out games with missing data
    df = df.dropna()

    # Select features
    feature_cols = [col for col in df.columns if col.endswith('_L3') or col.endswith('_L5')]

    # Add home/away indicator
    df['is_home'] = (df['home_away'] == 'home').astype(int)
    feature_cols.append('is_home')

    print(f"\nUsing {len(feature_cols)} features:")
    for col in feature_cols[:10]:
        print(f"  - {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")

    # Prepare data
    X = df[feature_cols].values
    y = df['total'].values

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Train model
    if model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"\nTraining {model_type} model with time series CV...")

    # Cross-validation
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        cv_scores.append({'fold': fold, 'mae': mae, 'rmse': rmse})

        print(f"  Fold {fold}: MAE={mae:.2f} points, RMSE={rmse:.2f} points")

    avg_mae = np.mean([s['mae'] for s in cv_scores])
    avg_rmse = np.mean([s['rmse'] for s in cv_scores])
    print(f"\n✓ Average CV MAE: {avg_mae:.2f} points")
    print(f"✓ Average CV RMSE: {avg_rmse:.2f} points")

    # Train final model on all data
    print("\nTraining final model on all data...")
    model.fit(X, y)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_type}_totals.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols,
            'cv_mae': avg_mae,
            'cv_rmse': avg_rmse
        }, f)

    print(f"\n✓ Saved model to {model_path}")

    # Feature importance (if applicable)
    if hasattr(model, 'coef_'):
        print("\nTop 10 features by coefficient:")
        coefs = pd.DataFrame({
            'feature': feature_cols,
            'coef': model.coef_
        }).sort_values('coef', key=abs, ascending=False)
        print(coefs.head(10).to_string(index=False))

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NFL totals model')
    parser.add_argument('--input', default='data/nfl/processed/team_features.csv', help='Team features CSV')
    parser.add_argument('--model-type', default='ridge', choices=['ridge', 'lasso', 'rf', 'gbm'], help='Model type')
    parser.add_argument('--output-dir', default='data/nfl/models', help='Output directory')

    args = parser.parse_args()

    train_totals_model(args.input, args.model_type, args.output_dir)
