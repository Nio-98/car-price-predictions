"""
model_comparison.py

Trenira i upoređuje više regresionih modela na ISTOM train/test skupu,
kako bi poređenje bilo fer.

Modeli:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import build_preprocessor, get_feature_target_split

RANDOM_STATE = 42


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=12),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=14
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def compare_models(data_path="data/cars_features.csv"):
    df = pd.read_csv(data_path)
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []

    for name, model in get_models().items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred)
        metrics["Model"] = name
        results.append(metrics)

        print(f"{name:20s} -> MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | R2: {metrics['R2']:.4f}")

    results_df = pd.DataFrame(results)[["Model", "MAE", "MSE", "RMSE", "R2"]]
    results_df = results_df.sort_values("MAE").reset_index(drop=True)

    print("\n=== Poređenje modela (sortirano po MAE) ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("models/model_comparison_results.csv", index=False)
    print("\nRezultati poređenja sačuvani u: models/model_comparison_results.csv")

    return results_df


if __name__ == "__main__":
    compare_models()
