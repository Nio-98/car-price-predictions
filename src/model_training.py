"""
model_training.py

Trenira finalni model za predviđanje cene polovnog automobila i čuva ga
u fajl (.joblib), zajedno sa test skupom koji koristi model_evaluation.py.

Finalni model: Random Forest Regressor.
Izbor je obrazložen u model_comparison.py / README.md - Random Forest je
dao najnižu MAE i najviši R2 od svih testiranih modela.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import build_preprocessor, get_feature_target_split

RANDOM_STATE = 42

FINAL_MODEL_PARAMS = dict(
    n_estimators=100,
    max_depth=14,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def train_final_model(
    data_path="data/cars_features.csv",
    model_output_path="models/car_price_model.joblib",
    test_data_output_path="models/test_data.joblib",
):
    df = pd.read_csv(data_path)
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(**FINAL_MODEL_PARAMS)),
    ])

    print("Treniranje finalnog modela (Random Forest Regressor)...")
    pipeline.fit(X_train, y_train)
    print("Treniranje završeno.")

    joblib.dump(pipeline, model_output_path)
    print(f"Model sačuvan u: {model_output_path}")

    # Čuvamo isti test skup da bi model_evaluation.py evaluirao model
    # na potpuno istim podacima koji NISU korišćeni u treningu
    joblib.dump((X_test, y_test), test_data_output_path)
    print(f"Test skup sačuvan u: {test_data_output_path}")

    return pipeline


if __name__ == "__main__":
    train_final_model()
