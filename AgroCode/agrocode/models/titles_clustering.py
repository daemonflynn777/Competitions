from sklearn.cluster import KMeans
import numpy as np
from typing import List

class TitlesClustering():
    def __init__(self, embeddings: List[List[float]], n_clusters: int = 21, init: str = "k-means++",
                 n_init: int = 35, max_iter: int = 1000, random_state: int = 666, algorithm: str = "auto"):
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.algorithm = algorithm

        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            algorithm=self.algorithm,
        )

    def get_fit_data(self) -> np.array:
        return np.array(self.embeddings)

    def run(self) -> List[int]:
        fit_data = self.get_fit_data()
        model_fit = self.model.fit(fit_data)
        classes = model_fit.labels_.tolist()
        return classes
