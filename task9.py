from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris_dataset = load_iris()

print("\nIRIS TARGET NAMES:\n", iris_dataset.target_names)

for i in range(len(iris_dataset.target_names)):
    print("[{0}]: [{1}]".format(i, iris_dataset.target_names[i]))

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"],
    iris_dataset["target"],
    random_state=0
)

# Create KNN classifier
kn = KNeighborsClassifier(n_neighbors=1)

# Train the model
kn.fit(X_train, y_train)

# Predict a new flower
x_new = np.array([[5, 2.9, 1, 0.2]])

print("\nX NEW:\n", x_new)

prediction = kn.predict(x_new)

print("\nPredicted target value:", prediction)
print("Predicted flower name:", iris_dataset["target_names"][prediction])

# Predict all test samples
for i in range(len(X_test)):
    x_new = np.array([X_test[i]])
    prediction = kn.predict(x_new)

    print("\nActual: {0} {1}, Predicted: {2} {3}".format(
        y_test[i],
        iris_dataset["target_names"][y_test[i]],
        prediction[0],
        iris_dataset["target_names"][prediction[0]]
    ))

# Calculate accuracy
print("\nTEST SCORE [ACCURACY]: {:.2f}".format(
    kn.score(X_test, y_test)
))
