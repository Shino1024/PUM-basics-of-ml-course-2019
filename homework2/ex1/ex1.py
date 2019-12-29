from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
from PIL import Image
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


COLORS = {
    "red": np.array([237, 28, 36]),
    "green": np.array([34, 177, 76]),
    "blue": np.array([0, 162, 232]),
    "yellow": np.array([255, 242, 0])
}


LABELS = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 3
}


FILENAMES = [
    "data" + str(i) + ".png" for i in range(3)
]


def dilute_data(data):
    diluted_image_data = []
    for coordinates in data:
        attenuation = np.array([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)])
        diluted_image_data.append(coordinates + attenuation)
    return diluted_image_data


def get_datasets():
    result = {}
    for filename in FILENAMES:
        image_result = []
        image_labels = []
        image_data = np.array(Image.open("../assets/" + filename).convert("RGB"))
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                pixel = image_data[i][j]
                for color in COLORS:
                    if np.all(pixel == COLORS[color]):
                        image_result.append([i, j])
                        image_labels.append(LABELS[color])
        diluted_image_result = np.array(dilute_data(image_result))
        image_labels = np.array(image_labels)
        result[filename] = {"model": diluted_image_result, "labels": image_labels}
    return result


def print_data(result):
    print(result)


def run():
    knns = [
        KNeighborsClassifier(algorithm="brute", n_neighbors=1, weights="uniform", metric="mahalanobis", n_jobs=1000),
        KNeighborsClassifier(algorithm="brute",  n_neighbors=7, weights="distance", metric="mahalanobis", n_jobs=1000),
        KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean", n_jobs=1000),
        KNeighborsClassifier(n_neighbors=7, weights="uniform", metric="euclidean", n_jobs=1000),
        KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean", n_jobs=1000),
    ]
    datasets = get_datasets()
    # print(datasets)

    h = 0.1
    for filename in FILENAMES:
        for knn in knns:
            if knn.metric == "mahalanobis":
                print(datasets[filename]["model"].shape)
                print(np.cov(datasets[filename]["model"]).shape)
                knn.metric_params = {"V": np.cov(datasets[filename]["model"])}
            # print_data(knn.fit(datasets[filename]["model"], datasets[filename]["labels"]))
            X = datasets[filename]["model"]
            y = datasets[filename]["labels"]
            knn.fit(X, y)

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            print("a")
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
            print("b")

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=ListedColormap(["#DC0B13", "#11A03B", "#0091D7", "#EEE100"]))

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#ED1C24", "#22B14C", "#00A2E8", "#FFF200"]),
                        edgecolor='k', s=20)
            print("c")
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("KNearestNeighbors (k = %i, weights = '%s', metric = '%s')"
                      % (knn.n_neighbors, knn.weights, knn.metric))
            plt.show()


if __name__ == "__main__":
    run()
