import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data from CSV file
data = pd.read_csv('credit_data.csv')

# Clean data by removing missing values
data.dropna(inplace=True)

# Split data into features and labels
features = data[['loan_size', 'interest_rate', 'borrower_income', 'debt_to_income',
                 'num_of_accounts', 'derogatory_marks', 'total_debt']]
labels = data['loan_status']

# Scale features to have zero mean and unit variance
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

# Build a logistic regression model for credit risk prediction
model = LogisticRegression()

# Train the model on the training set
model.fit(train_features, train_labels)

# Predict labels for the testing set
predictions = model.predict(test_features)

# Evaluate the model's accuracy and confusion matrix
accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
print('Accuracy:', accuracy)
print('Confusion Matrix:', conf_matrix)