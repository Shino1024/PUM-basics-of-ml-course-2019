import random
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DIMENSIONS = [
    3,
    4,
    5,
    7,
    13
]

POINTS = 50000
EDGE_POINTS = 10

RADIUS = 1
SIDE = 2 * RADIUS


def generate_usual_point(dimension):
    point = [random.uniform(-1.0, 1.0) for _ in range(dimension)]
    if sum(map(lambda coordinate: coordinate ** 2, point)) <= RADIUS ** 2:
        return point, [0.0, 1.0, 0.0]
    else:
        return point, [0.0, 0.0, 1.0]


def corner_points(dimension):
    return [([float(char) for char in map(lambda x: "-1" if x == "0" else x, list(np.binary_repr(d, width=dimension)))], [1.0, 0.0, 0.0]) for d in range(2 ** dimension)]


def side_points(dimension):
    return [
        (random.sample([random.choice([-1.0, 1.0]) for _ in range(dimension - 1)] + [random.uniform(-1.0, 1.0)], dimension), [1.0, 1.0, 0.0])
        for _ in range(EDGE_POINTS * dimension * (2 ** (dimension - 1)))
    ]


def generate_points():
    dimension_data = {dimension: [generate_usual_point(dimension)
                                  for _ in range(POINTS)] + side_points(dimension) + corner_points(dimension)
                      for dimension in DIMENSIONS}
    return dimension_data


def run_pca(dimension_data):
    pca_result = {2: {}, 3: {}}
    for dimension in DIMENSIONS:
        pca2 = PCA(n_components=2)
        points = dimension_data[dimension]
        pca_result[2][dimension] = pca2.fit_transform(points).tolist()
        pca3 = PCA(n_components=3)
        pca_result[3][dimension] = pca3.fit_transform(points).tolist()
    return pca_result


def run():
    dimension_data = generate_points()
    points = {dimension: [point[0] for point in dimension_data[dimension]] for dimension in DIMENSIONS}
    colors = {dimension: [point[1] for point in dimension_data[dimension]] for dimension in DIMENSIONS}
    pca_result = run_pca(points)
    for d in [2, 3]:
        for dimension in DIMENSIONS:
            plt.title("PCA: " + str(dimension) + "D -> " + str(d) + "D")
            plt.scatter(*zip(*pca_result[d][dimension]), color=colors[dimension])
            plt.show()

    # print(json.dumps(run_pca(points), indent=2))
    # print(json.dumps(dimension_data, indent=2))


if __name__ == "__main__":
    run()
