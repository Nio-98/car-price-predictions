import pandas as pd


def clean_data(input_path, output_path):
    # Učitavanje podataka
    df = pd.read_csv(input_path)

    print("Originalne dimenzije:", df.shape)

    # Uklanjanje duplikata
    duplicates = df.duplicated().sum()
    print("Broj duplikata:", duplicates)

    df = df.drop_duplicates()

    # Standardizacija naziva kolona
    df.columns = df.columns.str.strip()

    # Pretvaranje numeričkih kolona u numerički tip
    numeric_columns = [
        "priceUSD",
        "year",
        "mileage(kilometers)",
        "volume(cm3)"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Uklanjanje redova bez ciljne promenljive
    df = df.dropna(subset=["priceUSD"])

    # Uklanjanje nerealnih cena
    df = df[
        (df["priceUSD"] > 100) &
        (df["priceUSD"] < 200000)
    ]

    # Uklanjanje nerealnih godina
    df = df[
        (df["year"] >= 1980) &
        (df["year"] <= 2026)
    ]

    # Uklanjanje nerealne kilometraže
    df = df[
        (df["mileage(kilometers)"] >= 0) &
        (df["mileage(kilometers)"] <= 1000000)
    ]

    # Zapremina motora ne može biti negativna
    df.loc[
        df["volume(cm3)"] <= 0,
        "volume(cm3)"
    ] = pd.NA

    # Standardizacija tekstualnih kolona
    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment"
    ]

    for column in categorical_columns:
        df[column] = df[column].astype("string").str.strip().str.lower()

    # Popunjavanje missing vrednosti za kategorijske kolone
    for column in categorical_columns:
        df[column] = df[column].fillna("unknown")

    print("Dimenzije nakon čišćenja:", df.shape)

    print("\nNedostajuće vrednosti nakon čišćenja:")
    print(df.isnull().sum())

    # Čuvanje očišćenog dataseta
    df.to_csv(output_path, index=False)

    print(f"\nOčišćeni podaci sačuvani su u: {output_path}")


if __name__ == "__main__":
    clean_data(
        "data/cars.csv",
        "data/cars_clean.csv"
    )