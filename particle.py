"""Particle component for Particle Swarm Oprimization technique
"""

import numpy as np

from kmeans import KMeans, calc_sse


def quantization_error(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray) -> float:
    error = 0.0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)[0]
        dist = np.linalg.norm(data[idx] - c, axis=1).sum()
        dist /= len(idx)
        error += dist
    error /= len(centroids)
    return error


class Particle:
    """[summary]

    """

    def __init__(self,
                 n_cluster: int,
                 data: np.ndarray,
                 use_kmeans: bool = False,
                 w: float = 0.9,
                 c1: float = 0.5,
                 c2: float = 0.3):
        index = np.random.choice(list(range(len(data))), n_cluster)
        self.centroids = data[index].copy()
        if use_kmeans:
            kmeans = KMeans(n_cluster=n_cluster, init_pp=False)
            kmeans.fit(data)
            self.centroids = kmeans.centroid.copy()
        self.best_position = self.centroids.copy()
        self.best_score = quantization_error(self.centroids, self._predict(data), data)
        self.best_sse = calc_sse(self.centroids, self._predict(data), data)
        self.velocity = np.zeros_like(self.centroids)
        self._w = w
        self._c1 = c1
        self._c2 = c2

    def update(self, gbest_position: np.ndarray, data: np.ndarray):
        """Update particle's velocity and centroids
        
        Parameters
        ----------
        gbest_position : np.ndarray
        data : np.ndarray
        
        """
        self._update_velocity(gbest_position)
        self._update_centroids(data)

    def _update_velocity(self, gbest_position: np.ndarray):
        """Update velocity based on old value, cognitive component, and social component
        """

        v_old = self._w * self.velocity
        cognitive_component = self._c1 * np.random.random() * (self.best_position - self.centroids)
        social_component = self._c2 * np.random.random() * (gbest_position - self.centroids)
        self.velocity = v_old + cognitive_component + social_component

    def _update_centroids(self, data: np.ndarray):
        self.centroids = self.centroids + self.velocity
        new_score = quantization_error(self.centroids, self._predict(data), data)
        sse = calc_sse(self.centroids, self._predict(data), data)
        self.best_sse = min(sse, self.best_sse)
        if new_score < self.best_score:
            self.best_score = new_score
            self.best_position = self.centroids.copy()

    def _predict(self, data: np.ndarray) -> np.ndarray:
        """Predict new data's cluster using minimum distance to centroid
        """
        distance = self._calc_distance(data)
        cluster = self._assign_cluster(distance)
        return cluster

    def _calc_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate distance between data and centroids
        """
        distances = []
        for c in self.centroids:
            distance = np.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = np.transpose(distances)
        return distances

    def _assign_cluster(self, distance: np.ndarray) -> np.ndarray:
        """Assign cluster to data based on minimum distance to centroids
        """
        cluster = np.argmin(distance, axis=1)
        return cluster


if __name__ == "__main__":
    pass
