import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\ASUS\url\corrected_preprocessed_urls.csv")  # Replace with your file path

# Drop unnecessary columns (e.g., 'url')
data = data.drop(columns=['url'])

# Encode the target variable ('status')
label_encoder = LabelEncoder()
data['status'] = label_encoder.fit_transform(data['status'])  # Legitimate -> 1, Phishing -> 0

# Split the data into features (X) and target (y)
X = data.drop(columns=['status'])
y = data['status']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set up the SVM model and hyperparameter grid
svm = SVC()
param_grid = {
    'C': [0.1, 1, 10],           # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['rbf', 'linear']  # Kernel types
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best hyperparameters and model evaluation on test data
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

# Output results
print("Best Hyperparameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save both the trained model and the scaler
joblib.dump(best_svm, 'svm_phishing_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")
