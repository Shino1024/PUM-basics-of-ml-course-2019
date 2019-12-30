import random
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import davies_bouldin_score
import statistics
import matplotlib.pyplot as plt

CENTERS = 9
POINTS = [
    100,
    300,
    500,
    1000,
    # 3000,
    # 5000
]
K_MEANS_MAX = 20
ITERATIONS = 10
KMEANS_INITIALIZATION = [
    "random",
    "forgy",
    "random_partition",
    "k-means++"
]


def prepare_dataset():
    dataset = {}
    for points in POINTS:
        test_case = []
        for i in range(CENTERS):
            x = (i % 3) + 1
            y = (i // 3) + 1
            test_case.extend([
                (random.uniform(x - 0.3, x + 0.3), random.uniform(y - 0.3, y + 0.3)) for _ in range(points)
            ])
        dataset[points] = test_case
    return dataset


def point_in_circle(point, x, y, radius):
    return sum([(point[0] - x) ** 2, (point[1] - y) ** 2]) < radius ** 2


def prepare_spoiled_dataset():
    dataset = {}
    for points in POINTS:
        test_case = []
        test_case.extend(filter(lambda point: point_in_circle(point, 1, 1, 0.5), [
            (random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point, 2, 1, 0.3), [
            (random.uniform(1.7, 2.3), random.uniform(0.7, 1.3)) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point, 3, 1, 0.3), [
            (random.uniform(2.7, 3.3), random.uniform(0.7, 1.3)) for _ in range(points * 3)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point, 1, 2, 0.3), [
            (random.uniform(0.7, 1.3), random.uniform(1.7, 2.3)) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point, 2.25, 2, 0.3), [
            (random.uniform(1.95, 2.55), random.uniform(1.7, 2.3)) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point, 2.75, 2, 0.3), [
            (random.uniform(2.45, 3.05), random.uniform(1.7, 2.3)) for _ in range(points)
        ]))
        test_case.extend([
            (random.uniform(0.85, 1.15), random.uniform(2.6, 3.4)) for _ in range(points)
        ])
        test_case.extend(filter(lambda point: point_in_circle(point, 2, 8, 0.3), [
            (random.uniform(1.7, 2.3), random.uniform(7.7, 8.3)) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point, 3, 3, 0.3), [
            (random.uniform(2.7, 3.3), random.uniform(2.7, 3.3)) for _ in range(points)
        ]))
        dataset[points] = test_case
    return dataset


def get_random_centroids(dataset, no_centroids):
    return random.sample(list(dataset), no_centroids)


def show_dataset(dataset):
    for key in dataset:
        plt.scatter(*zip(*dataset[key]))
        plt.show()


def initialize_kmeans(initialization, dataset, no_clusters):
    kmeans = None

    if initialization == "random":
        uniform_distribution = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
        # print(uniform_distribution.shape)
        kmeans = KMeans(init=np.array(uniform_distribution), n_clusters=no_clusters, n_init=1, n_jobs=10)

    if initialization == "forgy":
        kmeans = KMeans(init="random", n_clusters=no_clusters, n_init=1, n_jobs=10)

    if initialization == "random_partition":
        partitions_centroids = get_random_centroids(dataset, no_clusters)
        kmeans = KMeans(init=np.array(partitions_centroids), n_clusters=no_clusters, n_init=1, n_jobs=10)

    if initialization == "k-means++":
        kmeans = KMeans(init="k-means++", n_clusters=no_clusters, n_init=1, n_jobs=10)

    return kmeans


def run_kmeans(dataset):
    kmeans_result = {i: {j: {name: {p: 0 for p in POINTS} for name in KMEANS_INITIALIZATION} for j in range(1, K_MEANS_MAX + 1)} for i in range(ITERATIONS)}
    for i in range(ITERATIONS):
        for j in range(1, K_MEANS_MAX + 1):
            for name in KMEANS_INITIALIZATION:
                # print("==========================")
                for p in POINTS:
                    # print(points[100].shape)
                    kmeans = initialize_kmeans(name, dataset[p], 9)
                    # print(list(zip(*points[p]))[0])
                    kmeans.fit(dataset[p])
                    kmeans_result[i][j][name][p] = davies_bouldin_score(dataset[p], kmeans.labels_)
                    # print(davies_bouldin_score(dataset[p], kmeans.labels_))
                    # print("--------")
                    # print(kmeans.cluster_centers_)
    return kmeans_result


def get_kmeans_iteration_mean(kmeans_result):
    means = {j: {name: {p: statistics.mean([kmeans_result[i][j][name][p] for i in range(ITERATIONS)]) for p in POINTS} for name in KMEANS_INITIALIZATION} for j in range(1, K_MEANS_MAX + 1)}
    return means


def get_kmeans_iteration_standard_deviation(kmeans_result):
    standard_deviations = {j: {name: {p: statistics.stdev([kmeans_result[i][j][name][p] for i in range(ITERATIONS)]) for p in POINTS} for name in KMEANS_INITIALIZATION} for j in range(1, K_MEANS_MAX + 1)}
    return standard_deviations


def run():
    dataset = prepare_dataset()
    spoiled_dataset = prepare_spoiled_dataset()
    # print(points[100])
    show_dataset(dataset)
    show_dataset(spoiled_dataset)
    kmeans_result = {"dataset": run_kmeans(dataset), "spoiled_dataset": run_kmeans(spoiled_dataset)}
    dataset_means = get_kmeans_iteration_mean(kmeans_result["dataset"])
    print(json.dumps(dataset_means, indent=2))
    spoiled_dataset_means = get_kmeans_iteration_mean(kmeans_result["spoiled_dataset"])
    print(json.dumps(spoiled_dataset_means, indent=2))
    dataset_standard_deviation = get_kmeans_iteration_standard_deviation(kmeans_result["dataset"])
    print(json.dumps(dataset_standard_deviation, indent=2))
    spoiled_dataset_standard_deviation = get_kmeans_iteration_standard_deviation(kmeans_result["spoiled_dataset"])
    print(json.dumps(spoiled_dataset_standard_deviation, indent=2))
    plot_data = {kmeans_initialization: {points: {} for points in POINTS} for kmeans_initialization in KMEANS_INITIALIZATION}
    for kmeans_initialization in KMEANS_INITIALIZATION:
        for points in POINTS:
            for k in range(1, K_MEANS_MAX + 1):
                plot_data[kmeans_initialization][points][k] = dataset_means[k][kmeans_initialization][points]
            plt.xlabel("k")
            plt.ylabel("Mean scores")
            plt.title("Mean scores - normal dataset - " + kmeans_initialization + ", " + str(points) + " points")
            plt.scatter(plot_data[kmeans_initialization][points].keys(), plot_data[kmeans_initialization][points].values())
            keys, values = list(plot_data[kmeans_initialization][points].keys()), list(plot_data[kmeans_initialization][points].values())
            for k in range(1, K_MEANS_MAX + 1):
                plot_data[kmeans_initialization][points][k] = dataset_standard_deviation[k][kmeans_initialization][points]
            plt.errorbar(keys, values, yerr=plot_data[kmeans_initialization][points].values())
            plt.show()
    for kmeans_initialization in KMEANS_INITIALIZATION:
        for points in POINTS:
            for k in range(1, K_MEANS_MAX + 1):
                plot_data[kmeans_initialization][points][k] = spoiled_dataset_means[k][kmeans_initialization][points]
            plt.xlabel("k")
            plt.ylabel("Mean scores")
            plt.title("Mean scores - spoiled dataset - " + kmeans_initialization + ", " + str(points) + " points")
            plt.scatter(plot_data[kmeans_initialization][points].keys(), plot_data[kmeans_initialization][points].values())
            keys, values = list(plot_data[kmeans_initialization][points].keys()), list(plot_data[kmeans_initialization][points].values())
            for k in range(1, K_MEANS_MAX + 1):
                plot_data[kmeans_initialization][points][k] = spoiled_dataset_standard_deviation[k][kmeans_initialization][points]
            plt.errorbar(keys, values, yerr=plot_data[kmeans_initialization][points].values())
            plt.show()


if __name__ == "__main__":
    run()
