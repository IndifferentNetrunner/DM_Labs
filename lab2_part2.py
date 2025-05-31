import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

# === Допоміжні функції ===
def euclidean(a, b):
    return np.linalg.norm(a - b)

def distance_matrix(data):
    n = len(data)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean(data[i], data[j])
            mat[i, j] = mat[j, i] = dist
    return mat

# === Ієрархічна кластеризація ===
class Hierarchical:
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.clusters = [[i] for i in range(self.n)]
        self.dists = distance_matrix(data)
        self.history = []

    def cluster(self, linkage='single'):
        self.clusters = [[i] for i in range(self.n)]
        self.history = []

        while len(self.clusters) > 1:
            i, j, dist = self._closest(linkage)
            self.history.append((min(self.clusters[i][0], self.clusters[j][0]),
                                 max(self.clusters[i][0], self.clusters[j][0]),
                                 dist, len(self.clusters[i]) + len(self.clusters[j])))
            merged = self.clusters.pop(j) + self.clusters.pop(i)
            self.clusters.append(merged)
        return self.history

    def _closest(self, linkage):
        best = float('inf')
        pair = (-1, -1)
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                c1, c2 = self.clusters[i], self.clusters[j]
                if linkage == 'single':
                    dist = min(self.dists[a][b] for a in c1 for b in c2)
                elif linkage == 'complete':
                    dist = max(self.dists[a][b] for a in c1 for b in c2)
                else:  # average
                    d = [self.dists[a][b] for a in c1 for b in c2]
                    dist = sum(d) / len(d)
                if dist < best:
                    best = dist
                    pair = (i, j)
        return (*pair, best)

    def get_clusters(self, num_clusters):
        parent = list(range(self.n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        current = self.n
        for a, b, _, _ in self.history:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                current -= 1
                if current == num_clusters:
                    break

        clusters = defaultdict(list)
        for i in range(self.n):
            clusters[find(i)].append(i)
        return list(clusters.values())

# === K-Means ===
def k_means(data, k, max_iter=100, tol=1e-4):
    centers = data[np.random.choice(len(data), k, replace=False)]
    labels = np.zeros(len(data), dtype=int)
    for _ in range(max_iter):
        old = centers.copy()
        for i, x in enumerate(data):
            labels[i] = np.argmin([euclidean(x, c) for c in centers])
        for i in range(k):
            points = data[labels == i]
            if len(points) > 0:
                centers[i] = points.mean(axis=0)
        if np.linalg.norm(centers - old) < tol:
            break
    return centers, labels

# === K-Medoids ===
def k_medoids(data, k, max_iter=100):
    medoids_idx = np.random.choice(len(data), k, replace=False)
    medoids = data[medoids_idx]
    labels = np.zeros(len(data), dtype=int)
    for _ in range(max_iter):
        for i, x in enumerate(data):
            labels[i] = np.argmin([euclidean(x, m) for m in medoids])
        for i in range(k):
            cluster = data[labels == i]
            if len(cluster) > 0:
                costs = [sum(euclidean(p, other) for other in cluster) for p in cluster]
                medoids[i] = cluster[np.argmin(costs)]
    return medoids, labels

# === DBSCAN ===
def dbscan(data, eps=2.0, min_pts=2):
    n = len(data)
    labels = np.full(n, -1)
    visited = np.zeros(n, bool)
    core = np.zeros(n, bool)

    neighbors = [np.where(np.linalg.norm(data - p, axis=1) <= eps)[0] for p in data]
    for i in range(n):
        if len(neighbors[i]) >= min_pts:
            core[i] = True

    cid = 0
    for i in range(n):
        if visited[i] or not core[i]:
            continue
        queue = deque([i])
        labels[i] = cid
        visited[i] = True
        while queue:
            q = queue.popleft()
            for nb in neighbors[q]:
                if not visited[nb]:
                    visited[nb] = True
                    labels[nb] = cid
                    if core[nb]:
                        queue.append(nb)
        cid += 1
    return labels

# === Візуалізація ===
def plot_clusters(data, labels, centers=None, title="Кластеризація"):
    plt.figure(figsize=(8, 6))
    unique = np.unique(labels)
    for u in unique:
        pts = data[labels == u]
        plt.scatter(pts[:, 0], pts[:, 1], s=80, label=f"Cluster {u}")
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', s=200, label='Centers')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_dendrogram(data, method='average'):
    Z = scipy_linkage(data, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(f"Дендрограма ({method.capitalize()} linkage)")
    plt.xlabel("Індекси точок")
    plt.ylabel("Відстань")
    plt.tight_layout()
    plt.show()

# === Дані та запуск ===
if __name__ == "__main__":
    data = np.array([
        [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6],
        [9, 11], [8, 2], [10, 2], [9, 3], [2, 2],
        [6, 8], [7, 9], [5, 5], [11, 2], [12, 3],
        [10, 10], [0, 1], [2, 3], [3, 4], [6, 5]
    ])
    k = 3
    eps = 2
    min_pts = 2

    print("K-Means:")
    centers, labels = k_means(data, k)
    plot_clusters(data, labels, centers, "K-Means")

    print("K-Medoids:")
    medoids, labels = k_medoids(data, k)
    plot_clusters(data, labels, medoids, "K-Medoids")

    print("DBSCAN:")
    labels = dbscan(data, eps, min_pts)
    plot_clusters(data, labels, title="DBSCAN")

    for method in ['single', 'complete', 'average']:
        print(f"Hierarchical ({method.capitalize()} Linkage):")
        hc = Hierarchical(data)
        hc.cluster(linkage=method)
        clusters = hc.get_clusters(num_clusters=3)

        labels = np.zeros(len(data), dtype=int)
        for idx, cluster in enumerate(clusters):
            for i in cluster:
                labels[i] = idx

        plot_clusters(data, labels, title=f"Hierarchical ({method.capitalize()} Linkage)")
        plot_dendrogram(data, method=method)