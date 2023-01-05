from agrocode.models import TitlesEmbeddings, TitlesClustering, ImageEmbeddings
from agrocode.utils import TitlesProcessing, load_yaml_safe, sort_images
import agrocode.config as cfg

from typing import Dict, List, Tuple
import fire
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Pipeline():
    def __init__(self, config_path: str):
        self.config_path = config_path

    def get_config_from_path(self) -> Dict:
        return load_yaml_safe(self.config_path)

    def make_titles_embeddings(self, titles_path: str) -> Tuple[pd.DataFrame, List[str]]:
        print("Cleaning titles")
        titles_processer = TitlesProcessing(titles_path)
        processed_titles = titles_processer.run()

        print("Creating titles embeddings")
        titles_embedder = TitlesEmbeddings(processed_titles[cfg.ITEM_COL].to_list())
        embedded_titles = titles_embedder.run()
        return processed_titles, embedded_titles

    def cluster_titles(self, embeddings: List[List[float]]):
        print("Clustering titles")
        titles_clusterer = TitlesClustering(embeddings)
        classes = titles_clusterer.run()
        return classes

    def make_images_embeddings(self, num_classes: int, use_saved_model: bool = False) -> tuple:
        image_embedder = ImageEmbeddings(num_epochs=12, transformations=[],  # "RandomRotation", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop"],
                                         batch_size=32, num_classes=num_classes, use_saved_model=use_saved_model)
        queries_image_ids, queries_image_embeddings, test_image_ids, test_image_embeddings = image_embedder.run()

        # train_image_embeddings_df = pd.DataFrame(
        #     data=[train_image_ids, train_image_embeddings], columns=["id", "embedding"]
        # )
        # test_image_embeddings_df = pd.DataFrame(
        #     data=[test_image_ids, test_image_embeddings], columns=["id", "embedding"]
        # )
        return np.array(queries_image_embeddings), np.array(test_image_embeddings)

    def find_similar_images(self, data_queries: np.array, data_test: np.array):
        queries = pd.read_csv(cfg.QUERIES_TITLES_PATH)
        test = pd.read_csv(cfg.TEST_TITLES_PATH)

        neigh = NearestNeighbors(n_neighbors=10, metric='cosine')
        neigh.fit(data_test)

        distances, idxs = neigh.kneighbors(data_queries, 10, return_distance=True)

        pred_data = pd.DataFrame()
        pred_data['score'] = distances.flatten()
        pred_data['database_idx'] = [test.idx.iloc[x] for x in idxs.flatten()]
        pred_data.loc[:, 'query_idx'] = np.repeat(queries.idx, 10).values
        return pred_data

    def save_predictions(self, df: pd.DataFrame):
        df.to_csv(cfg.SUBMISSION_PATH, index=False)

    def run(self):
        self.config = self.get_config_from_path()
        print(self.config["use_saved_model"])

        if not self.config["use_saved_model"]:
            titles_df, embedded_titles = self.make_titles_embeddings(cfg.TITLES_PATH)

            titles_classes = self.cluster_titles(embedded_titles.tolist())
            titles_df["class"] = titles_classes
            # print(titles_df.head(15))
            # print(titles_df["class"].value_counts())

            print("Sorting images into train and val")
            id_to_class_dict = {id: c for id, c in zip(titles_df["idx"].to_list(), titles_df["class"].to_list())}
            sort_images(id_to_class_dict, 4)

        emb_queries, emb_test = self.make_images_embeddings(num_classes=21, # =titles_df["class"].nunique(),
                                                            use_saved_model=self.config["use_saved_model"])

        df_similarity = self.find_similar_images(emb_queries, emb_test)

        self.save_predictions(df_similarity)

        print("Pipeline done!")


if __name__ == "__main__":
    fire.Fire(Pipeline)
