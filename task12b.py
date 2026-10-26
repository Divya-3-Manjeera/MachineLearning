import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

# Generate dataset
X, y = make_blobs(
    n_samples=600,
    centers=8,
    cluster_std=0.75,
    random_state=0
)

# Create BIRCH model
model = Birch(branching_factor=50, n_clusters=None, threshold=1.5)

# Train the model
model.fit(X)

# Predict clusters
pred = model.predict(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='rainbow')
plt.title("BIRCH Clustering")
plt.show()
