"""K-Means module, contain K-Means implementation inside KMeans class
"""

import numpy


def calc_sse(centroids: numpy.ndarray, labels: numpy.ndarray, data: numpy.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = numpy.where(labels == i)
        dist = numpy.sum((data[idx] - c)**2)
        distances += dist
    return distances


class KMeans:
    """K-Means clustering algorithm

        Attributes
        ----------
        n_cluster : int
            Num of cluster applied to data
        init_pp : bool
            Initialization method whether to use K-Means++ or not
            (the default is True, which use K-Means++)
        max_iter : int
            Max iteration to update centroid (the default is 300)
        tolerance : float
            Minimum centroid update difference value to stop iteration (the default is 1e-4)
        seed : int
            Seed number to use in random generator (the default is None)
        centroid : list
            List of centroid values
        SSE : float
            Sum squared error score
    """

    def __init__(
            self,
            n_cluster: int,
            init_pp: bool = True,
            max_iter: int = 300,
            tolerance: float = 1e-4,
            seed: int = None):
        """Instantiate K-Means object

        Parameters
        ----------
        n_cluster : int
            Num of cluster applied to data
        init_pp : bool, optional
            Initialization method whether to use K-Means++ or not
            (the default is True, which use K-Means++)
        max_iter : int, optional
            Max iteration to update centroid (the default is 100)
        tolerance : float, optional
            Minimum centroid update difference value to stop iteration (the default is 1e-4)
        seed : int, optional
            Seed number to use in random generator (the default is None)
        """

        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_pp = init_pp
        self.seed = seed
        self.centroid = None
        self.SSE = None

    def fit(self, data: numpy.ndarray):
        """Fit K-Means algorithm to given data

        Parameters
        ----------
        data : numpy.ndarray
            Data matrix to be fitted

        """
        self.centroid = self._init_centroid(data)
        for _ in range(self.max_iter):
            distance = self._calc_distance(data)
            cluster = self._assign_cluster(distance)
            new_centroid = self._update_centroid(data, cluster)
            diff = numpy.abs(self.centroid - new_centroid).mean()
            self.centroid = new_centroid

            if diff <= self.tolerance:
                break

        self.SSE = calc_sse(self.centroid, cluster, data)

    def predict(self, data: numpy.ndarray):
        """Predict new data's cluster using minimum distance to centroid

        Parameters
        ----------
        data : numpy.ndarray
            New data to be predicted

        """
        distance = self._calc_distance(data)
        # print(distance.shape)
        cluster = self._assign_cluster(distance)
        # print(cluster.shape)
        return cluster

    def _init_centroid(self, data: numpy.ndarray):
        """Initialize centroid using random method or KMeans++

        Parameters
        ----------
        data : numpy.ndarray
            Data matrix to sample from

        """
        if self.init_pp:
            numpy.random.seed(self.seed)
            centroid = [int(numpy.random.uniform()*len(data))]
            for _ in range(1, self.n_cluster):
                dist = []
                dist = [min([numpy.inner(data[c]-x, data[c]-x) for c in centroid])
                        for i, x in enumerate(data)]
                dist = numpy.array(dist)
                dist = dist / dist.sum()
                cumdist = numpy.cumsum(dist)

                prob = numpy.random.rand()
                for i, c in enumerate(cumdist):
                    if prob > c and i not in centroid:
                        centroid.append(i)
                        break
            centroid = numpy.array([data[c] for c in centroid])
        else:
            numpy.random.seed(self.seed)
            idx = numpy.random.choice(range(len(data)), size=(self.n_cluster))
            centroid = data[idx]
        # print(centroid)
        return centroid

    def _calc_distance(self, data: numpy.ndarray):
        """Calculate distance between data and centroids

        Parameters
        ----------
        data : numpy.ndarray
            Data which distance to be calculated

        """
        distances = []
        for c in self.centroid:
            distance = numpy.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = numpy.array(distances)
        distances = distances.T
        return distances

    def _assign_cluster(self, distance: numpy.ndarray):
        """Assign cluster to data based on minimum distance to centroids

        Parameters
        ----------
        distance : numpy.ndarray
            Distance from each data to each centroid

        """
        cluster = numpy.argmin(distance, axis=1)
        return cluster

    def _update_centroid(self, data: numpy.ndarray, cluster: numpy.ndarray):
        """Update centroid from means of each cluster's data

        Parameters
        ----------
        data : numpy.ndarray
            Data matrix to get mean from
        cluster : numpy.ndarray
            Cluster label for each data

        """
        centroids = []
        for i in range(self.n_cluster):
            idx = numpy.where(cluster == i)
            centroid = numpy.mean(data[idx], axis=0)
            centroids.append(centroid)
        centroids = numpy.array(centroids)
        return centroids


if __name__ == "__main__":

    pass
