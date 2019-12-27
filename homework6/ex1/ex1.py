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
    396 * 299,
    160 * 360,
    340 * 126,
    340 * 102,
    344 * 155,
    339 * 170,
    77 * 220,
    201 * 204,
    439 * 144,
    521 * 262
]


SCORING_METHODS = [
    "elbow",
    "silhouette",
    "gap_statistic"
]


def get_picture_data(remove_duplicates=False):
    picture_data = np.array(Image.open("../assets/balloon_forest.jpg").convert("RGB"))
    plt.imshow(picture_data)
    picture_shape = picture_data.shape
    picture_data = np.reshape(picture_data, (picture_data.shape[0] * picture_data.shape[1], picture_data.shape[2]))
    print(picture_data.shape)
    print(picture_shape)
    if remove_duplicates:
        pass
        # picture_data
    return picture_data, picture_shape


def calculate_distortion(dataset, model):
    return np.sum(np.min(cdist(dataset, model.cluster_centers_, "euclidean"), axis=1)) / dataset.shape[0]


def get_optimal_k_elbow(dataset):
    distortions = []
    for k in range(2, len(CHOSEN_POINTS) + 1):
        kmeans = KMeans(n_clusters=k)
        model = kmeans.fit(dataset)
        distortion = calculate_distortion(dataset, model)
        print(distortion)
        distortions.append(distortion)

    knee_locator = KneeLocator(range(1, len(distortions) + 1), distortions, curve="convex", direction="decreasing")
    print(knee_locator.knee)
    return knee_locator.knee


def get_optimal_k_silhouette(dataset):
    silhouettes = []
    for k in range(2, len(CHOSEN_POINTS) + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(dataset)
        print("Labels:")
        print(labels)
        print("Dataset:")
        print(dataset)
        silhouette = silhouette_score(dataset, labels)
        print(silhouette)
        silhouettes.append(silhouette)
    k = silhouettes.index(max(silhouettes))
    return k


def get_optimal_k_gap_statistic(dataset):
    optimal_k_generator = OptimalK(n_jobs=dataset.shape[0] ** 0.1)
    k = optimal_k_generator(dataset)
    return k


def get_optimal_k(dataset, method):
    print("Checking optimal k with the " + method + " method...")
    if method == SCORING_METHODS[0]:
        return get_optimal_k_elbow(dataset)
    elif method == SCORING_METHODS[1]:
        return get_optimal_k_silhouette(dataset)
    elif method == SCORING_METHODS[2]:
        return get_optimal_k_gap_statistic(dataset)
    pass


def choose_initial_points(dataset, k):
    chosen_points = random.sample(CHOSEN_POINTS, k)
    dataset_points = np.array([dataset[chosen_point] for chosen_point in chosen_points])
    return dataset_points


def run_kmeans(dataset, initial_points, shape):
    kmeans = KMeans(n_clusters=len(initial_points), init=initial_points, n_init=1)
    print(dataset)
    print(initial_points)
    result = kmeans.fit(dataset)
    print("Result0:")
    cluster_centers = kmeans.cluster_centers_
    print(cluster_centers)
    cluster_labels = kmeans.labels_
    print(cluster_labels)
    result = np.reshape(cluster_centers[cluster_labels], shape)
    print("Result:")
    print(result)
    print(result.shape)
    image = Image.fromarray(result, "RGB")
    image.show()
    # plt.imshow(result)
    # plt.show()
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
    # run_all(picture_data)


if __name__ == "__main__":
    run()
