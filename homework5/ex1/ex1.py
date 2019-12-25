import random
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import davies_bouldin_score
from pandas import DataFrame

CENTERS = 9
POINTS = [
    100,
    300,
    500,
    1000
]
K_MEANS_MAX = 20
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
                ((random.uniform(x - 0.3, x + 0.3), random.uniform(y - 0.3, y + 0.3)), i) for _ in range(points)
            ])
        dataset[points] = test_case
    return dataset


def point_in_circle(point, x, y, radius):
    return sum([(point[0] - x) ** 2, (point[1] - y) ** 2]) < radius ** 2


def prepare_spoiled_dataset():
    dataset = {}
    for points in POINTS:
        test_case = []
        test_case.extend(filter(lambda point: point_in_circle(point[0], 1, 1, 0.5), [
            ((random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)), 0) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point[0], 2, 1, 0.3), [
            ((random.uniform(1.7, 2.3), random.uniform(0.7, 1.3)), 1) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point[0], 3, 1, 0.3), [
            ((random.uniform(2.7, 3.3), random.uniform(0.7, 1.3)), 2) for _ in range(points * 3)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point[0], 1, 2, 0.3), [
            ((random.uniform(0.7, 1.3), random.uniform(1.7, 2.3)), 3) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point[0], 2.25, 2, 0.3), [
            ((random.uniform(1.95, 2.55), random.uniform(1.7, 2.3)), 4) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point[0], 2.75, 2, 0.3), [
            ((random.uniform(2.45, 3.05), random.uniform(1.7, 2.3)), 5) for _ in range(points)
        ]))
        test_case.extend([
            ((random.uniform(0.85, 1.15), random.uniform(2.6, 3.4)), 6) for _ in range(points)
        ])
        test_case.extend(filter(lambda point: point_in_circle(point[0], 2, 8, 0.3), [
            ((random.uniform(1.7, 2.3), random.uniform(7.7, 8.3)), 7) for _ in range(points)
        ]))
        test_case.extend(filter(lambda point: point_in_circle(point[0], 3, 3, 0.3), [
            ((random.uniform(2.7, 3.3), random.uniform(2.7, 3.3)), 8) for _ in range(points)
        ]))
        dataset[points] = test_case
    return dataset


def get_random_centroids(dataset, no_centroids):
    return random.shuffle(dataset)[0:no_centroids]


def show_dataset(dataset):
    for key in dataset:
        plt.scatter(*zip(*dataset[key]))
        plt.show()


def initialize_kmeans(initialization, dataset, no_clusters, no_iterations):
    kmeans = None

    if initialization == "random":
        uniform_distribution = [[[i, j] for j in range(1, 4)] for i in range(1, 4)]
        kmeans = KMeans(init=np.array(uniform_distribution), n_clusters=no_clusters)

    if initialization == "forgy":
        kmeans = KMeans(init="k-means++", n_clusters=no_clusters, n_init=no_iterations)

    if initialization == "random_partition":
        partitions_centroids = get_random_centroids(dataset, no_clusters)
        kmeans = KMeans(init=np.array(partitions_centroids), n_clusters=no_clusters)

    if initialization == "k-means++":
        kmeans = KMeans(init="k-means++", n_clusters=no_clusters, n_init=no_iterations)

    return kmeans


def run():
    dataset = prepare_dataset()
    spoiled_dataset = prepare_spoiled_dataset()
    points = {points: np.array(tuple(zip(*dataset[points]))[0]) for points in POINTS}
    classes = {points: np.array(tuple(zip(*dataset[points]))[1]) for points in POINTS}
    print(points[100])
    spoiled_points = {points: np.array(tuple(zip(*spoiled_dataset[points]))[0]) for points in POINTS}
    spoiled_classes = {points: np.array(tuple(zip(*spoiled_dataset[points]))[1]) for points in POINTS}
    show_dataset(points)
    show_dataset(spoiled_points)
    for name in KMEANS_INITIALIZATION:
        for p in POINTS:
            print(points[100].shape)
            kmeans = initialize_kmeans(name, points[p], 9, 10)
            print(list(zip(*points[p]))[0])
            data_frame = DataFrame({
                "x": list(zip(*points[p]))[0],
                "y": list(zip(*points[p]))[1]
            })
            kmeans.fit(data_frame)
            print(davies_bouldin_score(points[p], classes[p]))


if __name__ == "__main__":
    run()
