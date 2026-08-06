import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data

df = pd.read_csv("train-test.csv")
df["date"] = pd.to_datetime(df["date"])

# Data cleaning  //here instead of deleting the rows entirely we are replacing them with missing values

df.loc[df["weight"] <= 0, "weight"] = np.nan

# feature engineering

df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

# Route as a categorical feature
df["route"] = df["pickup"] + " -> " + df["delivery"]

# time base train, validation split 9 months train - 1 month validation

train_df = df[df["date"] < "2025-10-01"].copy()
valid_df = df[df["date"] >= "2025-10-01"].copy()

print(f"Training rows: {len(train_df):,}")
print(f"Validation rows: {len(valid_df):,}")

#Target

target = "posted_rate"

features = [
    "pickup",
    "delivery",
    "route",
    "pickup_lat",
    "pickup_lon",
    "distance",
    "equipment",
    "weight",
    "month",
    "day_of_week",
    "market_index",
    "quote_signal",
]

X_train = train_df[features]
y_train = train_df[target]

X_valid = valid_df[features]
y_valid = valid_df[target]

# Categorical features

categorical_features = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]

# Model

model = CatBoostRegressor(
    iterations = 1000,
    learning_rate = 0.05,
    depth = 8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
)

# Train

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100,
)

# validation predictions
model.save_model("catboost_route.cbm")

predictions = model.predict(X_valid)

# Error analysis
print()
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.get_feature_importance()
})

print(
    importance
    .sort_values("importance", ascending=False)
    .to_string(index=False)
)



# metrics

mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))
r2 = r2_score(y_valid, predictions)

print()
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")
