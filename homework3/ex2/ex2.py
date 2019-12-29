import random
import numpy as np
from PIL import Image
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import KernelPCA


RED = np.array([237, 28, 36])
GREEN = np.array([34, 177, 76])
BLUE = np.array([0, 162, 232])
YELLOW = np.array([255, 242, 0])


def prepare_dataset():
    dataset = {color: [] for color in ["red", "green", "blue", "yellow"]}
    handle = Image.open("../assets/PCA.png").convert("RGB")
    image = np.array(handle)
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (image[i][j] == RED).all():
                dataset["red"].append(((i, j), "red"))
            elif (image[i][j] == GREEN).all():
                dataset["green"].append(((i, j), "green"))
            elif (image[i][j] == BLUE).all():
                dataset["blue"].append(((i, j), "blue"))
            elif (image[i][j] == YELLOW).all():
                dataset["yellow"].append(((i, j), "yellow"))
    return dataset


def run_pca(data):
    pca2 = PCA(n_components=2)
    pca_data = pca2.fit_transform(data).tolist()
    # print(pca2.components_)
    # print(pca2.explained_variance_)
    # print(pca2.explained_variance_ratio_)
    return pca_data, pca2.components_


def run_cosine_pca(data, colors, centering):
    kernel_pca2_cosine = KernelPCA(n_components=2, kernel="cosine")
    if not centering:
        data = [[pixel[0] / 5 * ((pixel[0] / 321) ** 3), pixel[1] / 2 * ((pixel[1] / 435) ** 3)] for pixel in data]
    pca_cosine_data = kernel_pca2_cosine.fit_transform(data)
    plt.scatter(*zip(*pca_cosine_data), c=colors)
    plt.show()


def run_rbf_pca(data, colors, gamma):
    kernel_pca2_rbf = KernelPCA(n_components=2, kernel="rbf")
    kernel_pca2_rbf.gamma = gamma
    pca_rbf_data = kernel_pca2_rbf.fit_transform(data)
    plt.scatter(*zip(*pca_rbf_data), c=colors)
    plt.show()


def run_pca_with_kernel(data, colors):
    run_cosine_pca(data, colors, centering=False)
    run_cosine_pca(data, colors, centering=True)
    for gamma in [0.000001, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
        run_rbf_pca(data, colors, gamma)


def draw_pca_vectors(pca_components):
    figure, axes = plt.subplots(1)
    handle = Image.open("../assets/PCA.png").convert("RGB")
    image = np.array(handle)
    axes.imshow(handle)
    plt.quiver([image.shape[1] / 2], [image.shape[0] / 2], pca_components[:, 0], pca_components[:, 1], scale=10)
    plt.show()


def run():
    dataset = prepare_dataset()
    data = [point[0] for point in [item for sublist in dataset.values() for item in sublist]]
    colors = [point[1] for point in [item for sublist in dataset.values() for item in sublist]]
    pca_data, pca_components = run_pca(data)
    # print(json.dumps(pca_data, indent=2))
    plt.scatter(*zip(*pca_data), color=colors)
    draw_pca_vectors(pca_components)
    run_pca_with_kernel(data, colors)


if __name__ == "__main__":
    run()


"""
34 177 76
0 162 232
237 28 36
255 242 0
"""
