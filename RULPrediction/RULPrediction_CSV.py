import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the test dataset
data_path = "large_test_segmentation_outputs.csv"
data = pd.read_csv(data_path)

# Normalize RUL to percentages
data['severity'] = data['severity'].fillna('None')
data['defect_class'] = data.apply(
    lambda row: f"{row['defect_type']}_{row['severity']}" if row['severity'] != 'None' else row['defect_type'], axis=1
)
max_rul = data['RUL'].max()
data['RUL_percentage'] = (data['RUL'] / max_rul) * 100

features = data[['defect_class', 'mask_area', 'location_x', 'location_y']]
labels = data['RUL_percentage']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

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
pipeline.fit(X_train, y_train)

# Save the Model
joblib.dump(pipeline, 'rul_prediction_model.pkl')
print("Model saved as rul_prediction_model.pkl")

# Evaluate the Model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}%")
print(f"Mean Squared Error (MSE): {mse:.2f}%")
print(f"R-Squared (R²): {r2:.2f}")

# Predicting RUL for New Data
new_segmentation_outputs = [
    {'defect_type': 'Placking', 'severity': 'Low', 'mask_area': 1200, 'location': (25, 35)},
    {'defect_type': 'Chafing', 'severity': 'Low', 'mask_area': 900, 'location': (45, 55)}
]

# Feature extraction function
def extract_features(segmentation_outputs):
    features = []
    for output in segmentation_outputs:
        defect_type = output.get('defect_type', 'unknown')
        severity = output.get('severity', 'None')
        defect_size = output.get('mask_area', 0)
        location = output.get('location', (0, 0))

        # Combine defect type and severity to create unique classes
        if severity != 'None':
            combined_class = f"{defect_type}_{severity}"
        else:
            combined_class = defect_type

        features.append({
            'defect_class': combined_class,
            'mask_area': defect_size,
            'location_x': location[0],
            'location_y': location[1],
        })
    return pd.DataFrame(features)


new_features_df = extract_features(new_segmentation_outputs)
new_predictions = pipeline.predict(new_features_df)


print("\nPredicted RUL for New Data:")
for i, pred in enumerate(new_predictions):
    print(f"Sample {i+1}: Predicted RUL = {pred:.2f}%")
