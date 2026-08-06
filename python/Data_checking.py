import numpy as np
import pandas as pd


#loading the data
df = pd.read_csv("train-test.csv")
print("=" * 100)
print("Dataset overview")
print("=" * 100)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f" - {column}")

 #Data types

print("\n" + "=" * 100)
print("DATA TYPES")
print("=" * 100)

print(df.dtypes)


# Date range


print("\n" + "=" * 100)
print("DATE INFORMATION")
print("=" * 100)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print(f"Minimum date: {df['date'].min()}")
print(f"Maximum date: {df['date'].max()}")

print("\nRows per month:")
print(df["date"].dt.to_period("M").value_counts().sort_index())



# Missing values


print("\n" + "=" * 100)
print("MISSING VALUES")
print("=" * 100)

missing = df.isna().sum()
missing_percent = (missing / len(df) * 100).round(3)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent,
})

missing_report = missing_report[
    missing_report["missing_count"] > 0
].sort_values("missing_count", ascending=False)

if missing_report.empty:
    print("No missing values found.")
else:
    print(missing_report)



# Duplicate rows


print("\n" + "=" * 100)
print("DUPLICATES")
print("=" * 100)

print(f"Duplicate complete rows: {df.duplicated().sum():,}")

if "load_id" in df.columns:
    print(f"Duplicate load_id values: {df['load_id'].duplicated().sum():,}")



# Numeric statistics


numeric_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate",
]

print("\n" + "=" * 100)
print("NUMERIC STATISTICS")
print("=" * 100)

print(
    df[numeric_columns]
    .describe()
    .T
    .round(3)
)



# Invalid / suspicious values


print("\n" + "=" * 100)
print("INVALID / SUSPICIOUS VALUES")
print("=" * 100)

checks = {
    "distance <= 0": df["distance"] <= 0,
    "weight <= 0": df["weight"] <= 0,
    "market_index <= 0": df["market_index"] <= 0,
    "quote_signal <= 0": df["quote_signal"] <= 0,
    "posted_rate <= 0": df["posted_rate"] <= 0,
}

for name, mask in checks.items():
    print(f"{name:<25}: {mask.sum():,}")



# Categorical values


print("\n" + "=" * 100)
print("CATEGORICAL VALUES")
print("=" * 100)

for column in ["equipment"]:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))


print("\nUnique pickup locations:", df["pickup"].nunique())
print("Unique delivery locations:", df["delivery"].nunique())



# Coordinate consistency


print("\n" + "=" * 100)
print("COORDINATE CONSISTENCY")
print("=" * 100)

pickup_coordinates = (
    df.groupby("pickup")
    .agg(
        unique_lat=("pickup_lat", "nunique"),
        unique_lon=("pickup_lon", "nunique"),
    )
)

pickup_inconsistent = pickup_coordinates[
    (pickup_coordinates["unique_lat"] > 1)
    | (pickup_coordinates["unique_lon"] > 1)
]

print(f"Pickup cities with inconsistent coordinates: "
      f"{len(pickup_inconsistent)}")

if not pickup_inconsistent.empty:
    print(pickup_inconsistent)


delivery_coordinates = (
    df.groupby("delivery")
    .agg(
        unique_lat=("delivery_lat", "nunique"),
        unique_lon=("delivery_lon", "nunique"),
    )
)

delivery_inconsistent = delivery_coordinates[
    (delivery_coordinates["unique_lat"] > 1)
    | (delivery_coordinates["unique_lon"] > 1)
]

print(f"Delivery cities with inconsistent coordinates: "
      f"{len(delivery_inconsistent)}")

if not delivery_inconsistent.empty:
    print(delivery_inconsistent)



# Target distribution


print("\n" + "=" * 100)
print("TARGET: POSTED RATE")
print("=" * 100)

print(df["posted_rate"].describe().round(2))

print("\nHighest posted rates:")
print(
    df.nlargest(10, "posted_rate")[
        ["load_id", "pickup", "delivery", "distance",
         "equipment", "weight", "posted_rate"]
    ]
    .to_string(index=False)
)

print("\nLowest posted rates:")
print(
    df.nsmallest(10, "posted_rate")[
        ["load_id", "pickup", "delivery", "distance",
         "equipment", "weight", "posted_rate"]
    ]
    .to_string(index=False)
)


# Detailed data-quality investigation


print("\n" + "=" * 100)
print("DETAILED DATA QUALITY INVESTIGATION")
print("=" * 100)


#Invalid weights

invalid_weight = df[df["weight"] <= 0]

print("\nRows with weight <= 0:")
print(f"Count: {len(invalid_weight)}")

print(
    invalid_weight[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# Missing weights

missing_weight = df[df["weight"].isna()]

print("\nRows with missing weight:")
print(f"Count: {len(missing_weight)}")

print(
    missing_weight[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# Missing market index

missing_market = df[df["market_index"].isna()]

print("\nRows with missing market_index:")
print(f"Count: {len(missing_market)}")

print(
    missing_market[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# Feature relationships


print("\n" + "=" * 100)
print("FEATURE RELATIONSHIPS")
print("=" * 100)

numeric_columns = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate",
]

print("\nCorrelation matrix:")
print(df[numeric_columns].corr().round(3))
print("\nAverage posted rate by equipment:")
print(
    df.groupby("equipment")["posted_rate"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print("\nAverage posted rate by equipment and invalid weight status:")

df["weight_status"] = np.select(
    [
        df["weight"].isna(),
        df["weight"] <= 0,
    ],
    [
        "missing",
        "invalid",
    ],
    default="valid",
)

print(
    df.groupby("weight_status")["posted_rate"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

# Target and market trends over time


print("\n" + "=" * 100)
print("MONTHLY TRENDS")
print("=" * 100)

df["date"] = pd.to_datetime(df["date"])

monthly = (
    df.groupby(df["date"].dt.to_period("M"))
    .agg(
        posted_rate_mean=("posted_rate", "mean"),
        posted_rate_median=("posted_rate", "median"),
        market_index_mean=("market_index", "mean"),
        quote_signal_mean=("quote_signal", "mean"),
        distance_mean=("distance", "mean"),
        count=("load_id", "count"),
    )
)

print(monthly.round(2).to_string())
