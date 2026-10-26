import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv(
    r'C:/Users/divya/OneDrive/Desktop/Datasets/Datasets/PlayTennis.csv'
)

print(data.head())

# Convert categorical values into numbers
encoder = LabelEncoder()

for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

print(data.head())

# Select features and target
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target = 'Play Tennis'

X = data[features]
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=54
)

# Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict test results
pred = model.predict(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Predict for a new example
print("Prediction:", model.predict([[2, 1, 0, 1]]))
