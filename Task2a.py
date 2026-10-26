from sklearn.preprocessing import StandardScaler
import pandas
import numpy
dataframe = pandas.read_csv('C:\mllab\diabetes.csv', names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
array = dataframe.values
print(array)
X = array[:, 0:8]
Y = array[:, 8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
numpy.set_printoptions(precision = 3)
print(rescaledX[0:5,:])