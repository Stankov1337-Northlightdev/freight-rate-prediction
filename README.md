# freight-rate-prediction
Machine learning assessment for freight rate prediction using CatBoost.

The project uses CatBoost Regressor to predict posted_rate from market-related features.

Requirements
Python 3.10+
pandas
numpy
scikit-learn
catboost
matplotlib

Install dependencies:

pip install -r requirements.txt

How to run
1. Check the data
python src/data_check.py

This checks the main data quality issues in the development dataset.

2. Train the baseline model
python src/Training_program.py

This trains the initial CatBoost model and evaluates it on the October validation period.

3. Train the final model and generate validation predictions
python src/Final_training.py

The final model is trained on the available labeled data and saved for prediction.

This generates predictions for the 12,000 loads in data/validation.csv.

The final file is:

validation_predictions.csv

It contains:

load_id,predicted_rate

4. Generate December predictions
python src/december_training.py

This uses the trained final model to generate predictions for the fixed December 2025 scenario.

The script creates predictions for December 1–31, using:

Pickup: Lexington
Delivery: Fort Wayne
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000 lb

Only the date changes between rows.

The output is:

december_chart_inputs.csv
