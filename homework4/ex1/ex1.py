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
    15,
    50
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
        print(np.array(handle))
        image = []
        for row in np.array(handle):
            for pixel in row:
                image.append([pixel[0]])
        photos.append(image)
    return photos


def run_pca(dataset):
    dimension_data = {dimension: [] for dimension in DIMENSIONS}
    for dimension in DIMENSIONS:
        pca = PCA(n_components=dimension)
        for photo in dataset:
            print(len(photo))
            pca_data = pca.fit_transform(photo)
            dimension_data[dimension].append(pca_data)
    return dimension_data


def run():
    dataset = read_photos()
    # print(dataset[0][:50])
    pca_data = run_pca(dataset)
    print(json.dumps(pca_data, indent=2))


if __name__ == "__main__":
    run()


"""
34 177 76
0 162 232
237 28 36
255 242 0
"""
