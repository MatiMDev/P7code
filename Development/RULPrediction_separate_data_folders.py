import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load datasets
train_path = "datasets/RUL_data/RUL_data_train_adjusted.csv"
valid_path = "datasets/RUL_data/RUL_data_valid_adjusted.csv"
test_path = "datasets/RUL_data/RUL_data_test_adjusted.csv"

train_data = pd.read_csv(train_path)
valid_data = pd.read_csv(valid_path)
test_data = pd.read_csv(test_path)

# Combine training and validation data for training the model
train_data = pd.concat([train_data, valid_data], axis=0)

# Normalize RUL to percentages
def prepare_data(data, max_rul=None):
    data['severity'] = data['severity'].fillna('None')
    data['defect_class'] = data.apply(
        lambda row: f"{row['defect_type']}_{row['severity']}" if row['severity'] != 'None' else row['defect_type'], axis=1
    )
    if max_rul is None:
        max_rul = data['RUL'].max()
    data['RUL_percentage'] = (data['RUL'] / max_rul) * 100
    return data, max_rul

train_data, max_rul = prepare_data(train_data)
test_data, _ = prepare_data(test_data, max_rul)

# Extract features and labels
features_train = train_data[['defect_class', 'mask_area', 'location_x', 'location_y']]
labels_train = train_data['RUL_percentage']

features_test = test_data[['defect_class', 'mask_area', 'location_x', 'location_y']]
labels_test = test_data['RUL_percentage']

# Preprocessing: Encoding Categorical Variables and Scaling Numerical Features
categorical_features = ['defect_class']
numerical_features = ['mask_area', 'location_x', 'location_y']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Regression Model Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Training the Model
pipeline.fit(features_train, labels_train)

# Save the Model
joblib.dump(pipeline, 'rul_prediction_model.pkl')
print("Model saved as rul_prediction_model.pkl")

# Evaluate the Model
y_pred = pipeline.predict(features_test)
mae = mean_absolute_error(labels_test, y_pred)
mse = mean_squared_error(labels_test, y_pred)
r2 = r2_score(labels_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}%")
print(f"Mean Squared Error (MSE): {mse:.2f}%")
print(f"R-Squared (R²): {r2:.2f}")

# Predicting RUL for Test Data
test_data['Predicted_RUL_percentage'] = y_pred

# Save predictions for analysis
output_path = "datasets/RUL_data/test_predictions.csv"
test_data.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")

# Display a preview of predictions
print("\nPredicted RUL for Test Data (Sample):")
print(test_data[['defect_class', 'mask_area', 'location_x', 'location_y', 'RUL_percentage', 'Predicted_RUL_percentage']].head())
