import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Simulated Feature Extraction Function
def extract_features(segmentation_outputs):
    """
    Extract features from segmentation outputs.
    Classes with severity create separate sub-classes (e.g., Chafing_Low, Chafing_Medium).
    Defects without severity remain singular classes.

    Args:
        segmentation_outputs (list of dict): Outputs from the segmentation model.

    Returns:
        pd.DataFrame: Feature set for RUL prediction.
    """
    features = []
    for output in segmentation_outputs:
        defect_type = output.get('defect_type', 'unknown')
        severity = output.get('severity', None)
        defect_size = output.get('mask_area', 0)
        location = output.get('location', (0, 0))
        
        # Combine defect type and severity to create unique classes
        if severity and severity != 'None':
            combined_class = f"{defect_type}_{severity}"
        else:
            combined_class = defect_type

        features.append({
            'defect_class': combined_class,
            'defect_size': defect_size,
            'location_x': location[0],
            'location_y': location[1],
        })
    return pd.DataFrame(features)

# Simulated Segmentation Outputs
segmentation_outputs = [
    {'defect_type': 'Placking', 'severity': 'Low', 'mask_area': 1200, 'location': (30, 50)},
    {'defect_type': 'Cut Strands', 'severity': 'Medium', 'mask_area': 800, 'location': (60, 80)},
    {'defect_type': 'Chafing', 'severity': 'High', 'mask_area': 1500, 'location': (20, 40)},
    {'defect_type': 'Compression', 'severity': None, 'mask_area': 1800, 'location': (25, 30)},
    {'defect_type': 'Core Out', 'severity': None, 'mask_area': 1000, 'location': (50, 70)}
]

# Extract Features
features_df = extract_features(segmentation_outputs)
print("Extracted Features:")
print(features_df)

# Simulated RUL Labels (in cycles)
rul_labels = np.array([300, 450, 200, 150, 400])

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(features_df, rul_labels, test_size=0.2, random_state=42)

# Preprocessing: Encoding Categorical Variables and Scaling Numerical Features
categorical_features = ['defect_class']
numerical_features = ['defect_size', 'location_x', 'location_y']

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
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared (R²): {r2:.2f}")

# Predicting RUL for New Data
new_segmentation_outputs = [
    {'defect_type': 'Chafing', 'severity': 'Medium', 'mask_area': 1100, 'location': (25, 35)},
    {'defect_type': 'Compression', 'severity': None, 'mask_area': 1600, 'location': (45, 55)}
]

new_features_df = extract_features(new_segmentation_outputs)
new_predictions = pipeline.predict(new_features_df)

print("\nPredicted RUL for New Data:")
for i, pred in enumerate(new_predictions):
    print(f"Sample {i+1}: Predicted RUL = {pred:.2f} cycles")
