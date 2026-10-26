import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create dataset
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

# Plot original dataset
plt.scatter(x1, x2)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title("Original Dataset")
plt.show()

# Combine data points
X = np.array(list(zip(x1, x2)))

# Apply K-Means clustering
model = KMeans(n_clusters=3, random_state=0, n_init=10)
model.fit(X)

# Plot clusters
colors = ['blue', 'green', 'red']
markers = ['o', 'v', 's']

for i, label in enumerate(model.labels_):
    plt.plot(x1[i], x2[i], color=colors[label],
             marker=markers[label], linestyle='None')

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title("K-Means Clustering (K=3)")
plt.show()
