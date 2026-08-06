import numpy as np
import pandas as pd

from catboost import CatBoostRegressor



# Load data

train_df = pd.read_csv("train-test.csv")
validation_df = pd.read_csv("validation.csv")

print("=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

print(f"Development rows: {len(train_df):,}")
print(f"Validation rows:  {len(validation_df):,}")



#Data Cleaning


train_df["date"] = pd.to_datetime(train_df["date"])
validation_df["date"] = pd.to_datetime(validation_df["date"])

# Negative weight values are invalid.//here instead of deleting the rows entirely we are replacing them with missing values


train_df.loc[train_df["weight"] <= 0, "weight"] = np.nan
validation_df.loc[validation_df["weight"] <= 0, "weight"] = np.nan



# Feature engineering


for df in [train_df, validation_df]:

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    df["route"] = df["pickup"] + " -> " + df["delivery"]



# Features and target


target = "posted_rate"

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

X_train = train_df[features]
y_train = train_df[target]

X_validation = validation_df[features]



# Categorical features


categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]



# Final model


model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
)



# Training


print()
print("Training final model...")

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
)



# Saving the model


model.save_model("catboost_final.cbm")

# Predict validation data

predictions = model.predict(X_validation)



# Create submission file

submission = pd.DataFrame({
    "load_id": validation_df["load_id"],
    "predicted_rate": predictions,
})

submission.to_csv(
    "validation_predictions.csv",
    index=False
)



# Basic checks


print()
print("=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print(f"Predictions generated: {len(submission):,}")
print(f"Expected predictions:  12,000")

print()
print("Prediction statistics:")
print(submission["predicted_rate"].describe())

print()
print("First 10 predictions:")
print(submission.head(10))

print()
print("Output file:")
print("validation_predictions.csv")
