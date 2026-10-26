import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin',
             'bmi', 'pedigree', 'age', 'label']

pima = pd.read_csv(
    r'C:/Users/divya/OneDrive/Desktop/Datasets/Datasets/diabetes.csv',
    header=None,
    names=col_names
)

# Select input features and target
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp']
X = pima[feature_cols]
y = pima['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Create and train the Logistic Regression model
logreg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)

# Predict test results
y_pred = logreg.predict(X_test)

# Display confusion matrix
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

# Calculate accuracy and precision
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, zero_division=0))
