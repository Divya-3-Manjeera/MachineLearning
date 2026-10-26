import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Read dataset
df = pd.read_csv('C:/Users/divya/OneDrive/Desktop/Datasets/Datasets/50_Startups.csv')

# Convert categorical State column into numbers
df = pd.get_dummies(df, columns=['State'], dtype=int)

# Separate inputs and output
X = df.drop('Profit', axis=1)
y = df['Profit']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Evaluate the model
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
