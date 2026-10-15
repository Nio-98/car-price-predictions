"""
data_preprocessing.py

Definiše ciljnu promenljivu, numeričke i kategorijske kolone, i pravi
preprocessing pipeline (ColumnTransformer) koji se koristi u treniranju,
evaluaciji i poređenju modela.

Napomena o kolonama:
- "model" i "brand_model" imaju veliki broj jedinstvenih vrednosti
  (visoka kardinalnost). "make" već hvata glavninu informacije o brendu,
  a "brand_model" bi sa OneHotEncoder-om napravio ogroman broj kolona i
  rizik od overfittinga, pa ih izostavljamo iz ulaza za model i
  zadržavamo make/segment/ostale kategorijske karakteristike umesto njih.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "priceUSD"

NUMERIC_FEATURES = [
    "year",
    "mileage(kilometers)",
    "volume(cm3)",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters",
    "is_newer_car",
    "is_high_mileage",
]

CATEGORICAL_FEATURES = [
    "make",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_feature_target_split(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


if __name__ == "__main__":
    df = pd.read_csv("data/cars_features.csv")
    X, y = get_feature_target_split(df)

    print("Oblik X:", X.shape)
    print("Oblik y:", y.shape)
    print("\nNumeričke kolone:", NUMERIC_FEATURES)
    print("Kategorijske kolone:", CATEGORICAL_FEATURES)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    print("\nOblik nakon preprocesiranja:", X_transformed.shape)
