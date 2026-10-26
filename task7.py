import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv(
    r'C:/Users/divya/OneDrive/Desktop/Datasets/Datasets/cardata.csv',
    names=['buying', 'maint', 'doors', 'persons',
           'lug_boot', 'safety', 'class']
)

print(data.head())
print(data.info())

# Convert categorical columns into numerical values
for col in data.columns:
    data[col], _ = pd.factorize(data[col])

# Separate input features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Create and train Decision Tree
dtree = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=0
)
dtree.fit(X_train, y_train)

# Predict test results
y_pred = dtree.predict(X_test)

# Display confusion matrix
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

# Count misclassified samples
count_misclassified = (y_test != y_pred).sum()
print("Misclassified samples:", count_misclassified)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))
