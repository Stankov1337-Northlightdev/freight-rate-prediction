import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# Path
OUTPUT_PATH = "december_chart_inputs.csv"


# Load data and model
df = pd.read_csv("train-test.csv")
df["date"] = pd.to_datetime(df["date"])

model = CatBoostRegressor()
model.load_model("catboost_final.cbm")


# Find a real Lexington -> Fort Wayne row
route_data = df[
    (df["pickup"] == "Lexington") &
    (df["delivery"] == "Fort Wayne")
].copy()

if route_data.empty:
    raise ValueError("Lexington -> Fort Wayne route was not found")


# Take real coordinates from the dataset
pickup_lat = route_data["pickup_lat"].iloc[0]
pickup_lon = route_data["pickup_lon"].iloc[0]


# Use the latest available market values
market_index = df["market_index"].dropna().iloc[-1]
quote_signal = df["quote_signal"].dropna().iloc[-1]


# December dates
december = pd.DataFrame({
    "pickup": ["Lexington"] * 31,
    "delivery": ["Fort Wayne"] * 31,
    "pickup_lat": [pickup_lat] * 31,
    "pickup_lon": [pickup_lon] * 31,
    "distance": [360.0] * 31,
    "equipment": ["Dry Van"] * 31,
    "weight": [32000.0] * 31,
    "date": pd.date_range("2025-12-01", "2025-12-31"),
    "market_index": [market_index] * 31,
    "quote_signal": [quote_signal] * 31,
})


# Artificial route feature used during final training
december["route"] = (
    december["pickup"] + " -> " + december["delivery"]
)


# Date features
december["month"] = december["date"].dt.month
december["day_of_week"] = december["date"].dt.dayofweek


# EXACT feature order used by the final model
features = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "distance",
    "equipment",
    "weight",
    "month",
    "day_of_week",
    "market_index",
    "quote_signal",
    "route",
]


# Predict
december["predicted_rate"] = model.predict(december[features])


# Save required format
output_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]

december[output_columns].to_csv(
    OUTPUT_PATH,
    index=False
)


print("=" * 60)
print("DECEMBER PREDICTIONS")
print("=" * 60)

print(
    december[["date", "predicted_rate"]]
    .to_string(index=False)
)

print()
print(f"Saved to: {OUTPUT_PATH}")
