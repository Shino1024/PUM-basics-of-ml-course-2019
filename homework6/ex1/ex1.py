from PIL import Image
import numpy as np
from sklearn.metrics import silhouette_score
import json


# def score_


SCORING_METHODS = [
    "elbow",
    "silhouette",
    "gap_statistics"
]


def get_picture_data(remove_duplicates=False):
    picture_data = np.array(Image.open("../assets/globo-montseny.jpg").convert("RGB"))
    if remove_duplicates:
        pass
        # picture_data
    return picture_data


def get_optimal_k_elbow(dataset):
    pass


def get_optimal_k_silhouette(dataset):
    pass


def get_optimal_k_gap_statistics(dataset):
    pass


def get_optimal_k(dataset, method):
    if method == SCORING_METHODS[0]:
        return get_optimal_k_elbow(dataset)
    elif method == SCORING_METHODS[1]:
        return get_optimal_k_silhouette(dataset)
    elif method == SCORING_METHODS[2]:
        return get_optimal_k_gap_statistics(dataset)
    pass


def choose_initial_points(dataset, k):
    pass


def run_kmeans(dataset, initial_points):
    pass


def map_2d_pca(kmeans_result):
    pass


def draw_clusters(pca_map):
    pass


def draw_silhouette(pca_map, labels):
    silhouette = silhouette_score(pca_map, labels)


def run_all(dataset):
    for scoring_method in SCORING_METHODS:
        k = get_optimal_k(dataset, scoring_method)
        initial_points = choose_initial_points(dataset, k)
        kmeans_result = run_kmeans(dataset, initial_points)
        pca_map = map_2d_pca(kmeans_result)
        draw_clusters(pca_map)
        draw_silhouette(pca_map)


def run():
    picture_data = get_picture_data(False)
    run_all(picture_data)
    #
    picture_data = get_picture_data(True)
    run_all(picture_data)


if __name__ == "__main__":
    run()
