"""
feature_engineering.py

Kreira dodatne karakteristike na osnovu očišćenog dataseta polovnih automobila.

Nove karakteristike:
- car_age               : starost automobila (trenutna_godina - year)
- mileage_per_year       : prosečna kilometraža po godini starosti (uz zaštitu od deljenja nulom)
- engine_volume_liters   : zapremina motora u litrima (volume(cm3) / 1000)
- is_newer_car           : 1 ako je automobil star manje od 5 godina, inače 0
- is_high_mileage        : 1 ako je kilometraža veća od 200,000 km, inače 0
- brand_model            : kombinacija marke i modela (npr. "volkswagen_golf")
"""

import datetime
import pandas as pd

CURRENT_YEAR = datetime.datetime.now().year


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Starost automobila
    df["car_age"] = CURRENT_YEAR - df["year"]
    # Teoretski nemoguće (auto iz "budućnosti"), ali za svaki slučaj ne dozvoljavamo negativne vrednosti
    df["car_age"] = df["car_age"].clip(lower=0)

    # Prosečna kilometraža po godini starosti
    # +1 u imeniocu da izbegnemo deljenje nulom za potpuno nove automobile (car_age == 0)
    df["mileage_per_year"] = df["mileage(kilometers)"] / (df["car_age"] + 1)

    # Zapremina motora u litrima - lakše za tumačenje i za model
    df["engine_volume_liters"] = df["volume(cm3)"] / 1000

    # Indikator da li je automobil novijeg godišta (mlađi od 5 godina)
    df["is_newer_car"] = (df["car_age"] < 5).astype(int)

    # Indikator da li automobil ima veoma veliku kilometražu (preko 200,000 km)
    df["is_high_mileage"] = (df["mileage(kilometers)"] > 200_000).astype(int)

    # Kombinacija marke i modela - hvata specifičnu vrednost pojedinih modela
    # koja se gubi kada se make i model posmatraju odvojeno
    df["brand_model"] = df["make"].astype(str) + "_" + df["model"].astype(str)

    return df


def engineer_features(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print("Dimenzije pre inženjeringa karakteristika:", df.shape)

    df = add_features(df)

    print("Dimenzije nakon inženjeringa karakteristika:", df.shape)
    print("\nNove kolone:")
    new_cols = ["car_age", "mileage_per_year", "engine_volume_liters",
                "is_newer_car", "is_high_mileage", "brand_model"]
    print(df[new_cols].head())

    df.to_csv(output_path, index=False)
    print(f"\nDataset sa novim karakteristikama sačuvan u: {output_path}")

    return df


if __name__ == "__main__":
    engineer_features(
        "data/cars_clean.csv",
        "data/cars_features.csv"
    )
