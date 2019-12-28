import random
from PIL import Image
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    "red": np.array([237, 28, 36]),
    "blue": np.array([0, 162, 232])
}


C_COEFFICIENTS = [
    0.01,
    0.03,
    0.1,
    0.3,
    1,
    3,
    10,
    30,
    100,
    300,
    1000
]


SVC_KERNELS = [
    "rbf",
    "linear",
    "poly"
]


GAMMAS = np.logspace(-9, 3, 13)


def load_data():
    picture_data = np.array(Image.open("../assets/data.png").convert("RGB"))
    picture_shape = picture_data.shape
    pixels = []
    labels = []
    for i in range(picture_data.shape[0]):
        for j in range(picture_data.shape[1]):
            pixel = picture_data[i][j]
            if np.all(pixel == COLORS["red"]):
                pixels.append((i, j))
                labels.append(0)
            elif np.all(pixel == COLORS["blue"]):
                pixels.append((i, j))
                labels.append(1)
    return np.array(pixels), np.array(labels), picture_shape


def run_svm(dataset, labels, kernel, c, gamma="scale"):
    svc = None
    if kernel == "poly":
        svc = SVC(kernel=kernel, degree=3, C=c, gamma=gamma)
    else:
        svc = SVC(kernel=kernel, C=c, gamma=gamma)
    svc.fit(dataset, labels)
    # print(svc.support_vectors_)
    return svc


def pixel_in_appropriate_area(a, xx, pixel):
    return 1


def run():
    pixels, labels, shape = load_data()
    results = {svc_kernel: {c_coefficient: [] for c_coefficient in C_COEFFICIENTS} for svc_kernel in SVC_KERNELS}
    for c in C_COEFFICIENTS:
        svm_result = run_svm(pixels, labels, "linear", c)
        w = svm_result.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(0, 96)
        yy = a * xx - (svm_result.intercept_[0]) / w[1]
        margin = 1 / np.sqrt(np.sum(svm_result.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

# plot the line, the points, and the nearest vectors to the plane
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        plt.scatter(svm_result.support_vectors_[:, 0], svm_result.support_vectors_[:, 1],
                    s=80, facecolors='none')
        plt.scatter(pixels[:, 0], pixels[:, 1], c=list(map(lambda x: "#FF0000" if x == 0 else "#0000FF", labels)))

        plt.axis('tight')
        plt.show()
        wrongness_list = [
            pixel_in_appropriate_area(a, xx, pixel) for pixel in pixels
        ]
        wrongness = sum(wrongness_list) / len(wrongness_list)
        results["linear"][c] = [(margin, wrongness)]
        print("=============")

    for c in C_COEFFICIENTS:
        svm_result = run_svm(pixels, labels, "poly", c)
        results["poly"][c] = [svm_result]

    for c in C_COEFFICIENTS:
        results["rbf"][c] = {}
        for gamma in GAMMAS:
            svm_result = run_svm(pixels, labels, "rbf", c, gamma)
            print(svm_result.support_vectors_)
            results["rbf"][c][gamma] = svm_result

    # print(results["linear"])


if __name__ == "__main__":
    run()
