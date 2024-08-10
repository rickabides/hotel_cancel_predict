# Import necessary libraries
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Create a directory for saving plots if it doesn't exist
os.makedirs('training_plots', exist_ok=True)

# Load the CSV file
data = pd.read_csv('/Users/rbryce/Downloads/HotelReservations_trim_v2.csv')

# Prepare data by removing unnecessary columns
hotel_data = data.drop(columns=['BookingID', 'ArrivalDate', 'ArrivalYear', 'ArrivalMonth'])

# One-hot encode categorical variables
categorical_vars = ['TypeOfMealPlan', 'RoomTypeReserved', 'MarketSegmentType']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_vars = encoder.fit_transform(hotel_data[categorical_vars])
encoded_df = pd.DataFrame(encoded_vars, columns=encoder.get_feature_names_out(categorical_vars))

# Combine encoded variables with numerical data
numerical_vars = hotel_data.drop(columns=categorical_vars + ['BookingStatus']).select_dtypes(include=[np.number])
hotel_data_encoded = pd.concat([numerical_vars.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Create a new variable for total nights
hotel_data_encoded['TotalNights'] = hotel_data['NoOfWeekendNights'] + hotel_data['NoOfWeekNights']

# Define the target variable
y = (hotel_data['BookingStatus'] == 'Not_Canceled').astype(int)
X = hotel_data_encoded  # No need to drop any columns here

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit the Gradient Boosting model
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_train, y_train)

# Generate a datestamp
datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the trained model with the datestamp in the filename
model_filename = f'trained_gbm_model_{datestamp}.pkl'
joblib.dump(gbm_model, model_filename)

# Save the encoder
encoder_filename = f'encoder_gbm_{datestamp}.joblib'
joblib.dump(encoder, encoder_filename)

# Save the column names
col_filename = f'col_gbm_{datestamp}.joblib'
joblib.dump(X.columns.tolist(), col_filename)

# Make predictions on the test set
predictions = gbm_model.predict(X_test)
probabilities = gbm_model.predict_proba(X_test)[:, 1]

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc_value = roc_auc_score(y_test, probabilities)

# Print metrics
print("Confusion Matrix:\n", conf_matrix)
print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc_value:.4f}")

# Plot ROC curve
plt.figure(figsize=(10.24, 7.68), dpi=100)
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
plt.plot(fpr, tpr, label=f'AUC = {auc_value:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('training_plots/roc_curve.png')  # Save the ROC curve plot
plt.close()

# Feature Importance
feature_importance = gbm_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nTop 10 Feature Importances:")
print(importance_df.head(10))

# Plot Feature Importance
plt.figure(figsize=(10.24, 7.68), dpi=100)
importance_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('training_plots/feature_importance.png')  # Save the feature importance plot
plt.close()
