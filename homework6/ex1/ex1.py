from PIL import Image
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import json
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from gap_statistic.optimalK import OptimalK
import random
import matplotlib.pyplot as plt


CHOSEN_POINTS = [
    67 * 25,
    106 * 64,
    261 * 76,
    261 * 105,
    66 * 176,
    138 * 158,
    359 * 199,
    308 * 226,
    149 * 258,
    261 * 128,
    402 * 147
]


SCORING_METHODS = [
    "elbow",
    "silhouette",
    "gap_statistic"
]


def get_picture_data(remove_duplicates=False):
    picture_data = np.array(Image.open("../assets/balloon_forest.jpg").convert("RGB"))
    picture_shape = picture_data.shape
    picture_data = np.reshape(picture_data, (picture_data.shape[0] * picture_data.shape[1], picture_data.shape[2]))
    if remove_duplicates:
        picture_data = np.unique(picture_data)
        # picture_data
    return picture_data, picture_shape


def calculate_distortion(dataset, cluster_centers):
    return np.sum(np.min(cdist(dataset, cluster_centers, "euclidean"), axis=1)) / dataset.shape[0]


def get_optimal_k_elbow(dataset):
    distortions = []
    for k in range(2, len(CHOSEN_POINTS) + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataset)
        distortion = calculate_distortion(dataset, kmeans.cluster_centers_)
        distortions.append(distortion)

    knee_locator = KneeLocator(range(1, len(distortions) + 1), distortions, curve="convex", direction="decreasing")
    return knee_locator.knee


def get_optimal_k_silhouette(dataset):
    silhouettes = []
    for k in range(2, len(CHOSEN_POINTS) + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(dataset)
        silhouette = silhouette_score(dataset, labels, sample_size=2 ** 12)
        silhouettes.append(silhouette)
    k = silhouettes.index(max(silhouettes)) + 2
    return k


def get_optimal_k_gap_statistic(dataset):
    optimal_k_generator = OptimalK(n_jobs=dataset.shape[0] ** 0.3)
    print(dataset.shape)
    k = optimal_k_generator(dataset, cluster_array=np.arange(2, len(CHOSEN_POINTS) + 1))
    return k


def get_optimal_k(dataset, method):
    print("Checking optimal k with the " + method + " method...")
    k = 0
    if method == SCORING_METHODS[0]:
        k = get_optimal_k_elbow(dataset)
    elif method == SCORING_METHODS[1]:
        k = get_optimal_k_silhouette(dataset)
    elif method == SCORING_METHODS[2]:
        k = get_optimal_k_gap_statistic(dataset)
    print("Optimal k for the " + method + " method is " + k)
    return k


def choose_initial_points(dataset, k):
    chosen_points = random.sample(CHOSEN_POINTS, k)
    # return np.array(chosen_points)
    dataset_points = np.array([dataset[chosen_point] for chosen_point in chosen_points])
    return dataset_points


def run_kmeans(dataset, initial_points, shape):
    kmeans = KMeans(n_clusters=len(initial_points), init=initial_points, n_init=1)
    kmeans.fit(dataset)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    clustered_image = cluster_centers[cluster_labels].astype(np.int64)
    result = np.reshape(clustered_image, shape)
    plt.imshow(result)
    plt.show()
    plt.imshow(np.reshape(dataset, shape))
    return result


def map_2d_pca(kmeans_result):
    pass


def draw_clusters(pca_map):
    pass


def draw_silhouette(pca_map, labels):
    silhouette = silhouette_score(pca_map, labels)


def run_all(dataset, shape):
    for scoring_method in SCORING_METHODS:
        k = get_optimal_k(dataset, scoring_method)
        print("OPTIMAL K: ", k)
        initial_points = choose_initial_points(dataset, k)
        kmeans_result = run_kmeans(dataset, initial_points, shape)
        # pca_map = map_2d_pca(kmeans_result)
        # draw_clusters(pca_map)
        # draw_silhouette(pca_map, [])


def run():
    picture_data, picture_shape = get_picture_data(False)
    run_all(picture_data, picture_shape)
    # run_all(picture_data)
    #
    picture_data, picture_shape = get_picture_data(True)
    run_all(picture_data, picture_shape)


if __name__ == "__main__":
    run()
