import random
import numpy as np
from PIL import Image
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import KernelPCA
import glob


DIMENSIONS = [
    5,
    10,
    15,
    25
]


def draw_pca_vectors(pca_components):
    figure, axes = plt.subplots(1)
    handle = Image.open("../assets/PCA.png").convert("RGB")
    image = np.array(handle)
    axes.imshow(handle)
    plt.quiver([image.shape[1] / 2], [image.shape[0] / 2], pca_components[:, 0], pca_components[:, 1], scale=10)
    plt.show()


def read_photos():
    photos = []
    paths = []
    for path in glob.glob("../assets/faces/*.jpg"):
        paths.append(path)
    paths = sorted(paths)
    for path in paths:
        print(path)
        handle = Image.open(path).convert("LA")
        image = []
        for row in np.array(handle):
            for pixel in row:
                image.append(pixel[0])
        photos.append(image)
    return np.array(photos)


def run_pca(dataset):
    dimension_data = {}
    for dimension in DIMENSIONS:
        pca = PCA(n_components=dimension)
        print(dataset.shape)
        pca_data = pca.fit_transform(dataset)
        dimension_data[dimension] = np.array(pca_data)
    return dimension_data


def run():
    dataset = read_photos()
    # print(dataset[0][:50])
    pca_data = run_pca(dataset)
    for dimension in reversed(DIMENSIONS):
        plt.imshow(pca_data[dimension], cmap="gray")
        plt.show()
        # fig = plt.figure(figsize=(6,6))
        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        # ax = fig.add_subplot(8, 8, DIMENSIONS.index(dimension) + 1, xticks=[], yticks=[])
        # ax.imshow(pca_data[dimension], interpolation='nearest')
        # for i in range(dimension):
        #     ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        #     ax.imshow(pca_data[dimension][i], cmap=plt.cm.bone, interpolation='nearest')
    # plt.show()
    # fig, axes = plt.subplots(10, 10, figsize=(9, 9), subplot_kw={"xticks":[], "yticks":[]},
    #                          gridspec_kw = dict(hspace=0.01, wspace=0.01))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(faces.iloc[i].values.reshape(112, 92), cmap="gray")


if __name__ == "__main__":
    run()


"""
34 177 76
0 162 232
237 28 36
255 242 0
"""
