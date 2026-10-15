"""
model_evaluation.py

Učitava sačuvani model i test skup, izračunava regresione metrike
(MAE, MSE, RMSE, R2) i prikazuje nekoliko primera stvarna vs. predviđena cena.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model_path="models/car_price_model.joblib",
    test_data_path="models/test_data.joblib",
):
    pipeline = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("=== Rezultati evaluacije finalnog modela ===")
    print(f"MAE  (Mean Absolute Error):      {mae:,.2f} USD")
    print(f"MSE  (Mean Squared Error):       {mse:,.2f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:,.2f} USD")
    print(f"R2   (koeficijent determinacije): {r2:.4f}")

    print(
        f"\nTumačenje: u proseku, model greši oko {mae:,.0f}$ pri proceni cene "
        f"automobila (MAE), a objašnjava približno {r2 * 100:.1f}% varijanse cene "
        f"u test skupu (R2)."
    )

    # Primeri predviđanja
    comparison_df = pd.DataFrame({
        "stvarna_cena": y_test.values,
        "predvidjena_cena": np.round(y_pred, 0),
    })
    comparison_df["greska"] = (comparison_df["predvidjena_cena"] - comparison_df["stvarna_cena"]).abs()

    sample = comparison_df.sample(n=10, random_state=42).sort_values("greska")

    print("\n=== Primeri predviđanja (10 nasumičnih automobila iz test skupa) ===")
    print(sample.to_string(index=False))

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    evaluate_model()
