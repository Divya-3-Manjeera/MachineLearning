import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
col_names = [
    'pregnant', 'glucose', 'bp', 'skin', 'insulin',
    'bmi', 'pedigree', 'age', 'label'
]
pima = pd.read_csv(r'C:\mllab\diabetes.csv',header=None,names=col_names)
print(pima)
feature_cols = [
    'pregnant', 'insulin', 'bmi',
    'age', 'glucose', 'bp'
]
X = pima[feature_cols]
y = pima['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('R2 Score =', r2_score(y_test, y_pred))
print('Mean Squared Error =',
      mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error =',
      np.sqrt(mean_squared_error(y_test, y_pred)))