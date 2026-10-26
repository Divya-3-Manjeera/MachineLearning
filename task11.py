import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(
    r'C:/Users/divya/OneDrive/Desktop/Datasets/Datasets/diabetes.csv',
    names=names
)

# Split features and target
X = data.drop('class', axis=1)
y = data['class']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=66
)

# Create and train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10-fold cross-validation using ROC-AUC
scores = cross_val_score(model, X, y, cv=10, scoring='roc_auc')

print("\nAll AUC Scores:")
print(scores)

print("\nMean AUC Score:", scores.mean())
