import random
from PIL import Image
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import matplotlib


COLORS = {
    "red": np.array([237, 28, 36]),
    "blue": np.array([0, 162, 232])
}


C_COEFFICIENTS = [
    0.001,
    0.01,
    0.1,
    1,
    10
]


SVC_KERNELS = [
    "rbf",
    "linear",
    "poly"
]


GAMMAS = np.logspace(-5, 2, 10)


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
        svc = SVC(kernel=kernel, degree=3, C=c, gamma=gamma, probability=True)
    else:
        svc = SVC(kernel=kernel, C=c, gamma=gamma, probability=True)
    svc.fit(dataset, labels)
    # print(svc.support_vectors_)
    return svc


def pixel_in_appropriate_area_linear(a, xx, yy, pixel, classification):
    # print("a:", a)
    # print("xx:", xx)
    # print("yy:", yy)
    # print("pixel:", pixel)
    b = yy[0] - a * xx[0]
    # print("B:", b)
    if classification == 1:
        if pixel[1] >= a * pixel[0] + b:
            return 1
        elif pixel[1] <= a * pixel[0] + b:
            return 0
    elif classification == -1:
        if pixel[1] >= a * pixel[0] + b:
            return 0
        elif pixel[1] <= a * pixel[0] + b:
            return 1
    return 0


def calculate_data(svm_result, pixels, kernel, c, labels):
    plt.title("Classification with the kernel " + kernel + ", c = " + str(c))
    if kernel == "linear":
        w = svm_result.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(0, 64)
        yy = a * xx - (svm_result.intercept_[0]) / w[1]
        margin = 1 / np.sqrt(np.sum(svm_result.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin
        # plt.show()
        return a, xx, yy, margin
    elif kernel == "rbf":
        # plt.show()
        return 20
    elif kernel == "poly":
        return 50


def calculate_wrongness_linear(svm_result, a, xx, yy, pixels):
    wrong_pixels_count = {"red": 0, "blue": 0}
    pixels_count = {"red": 0, "blue": 0}
    for pixel in pixels:
        classification = svm_result.predict([pixel])
        if classification > 0:
            if not pixel_in_appropriate_area_linear(a, xx, yy, pixel, classification):
                wrong_pixels_count["red"] += 1
            pixels_count["red"] += 1
        elif classification == 0:
            if not pixel_in_appropriate_area_linear(a, xx, yy, pixel, classification):
                wrong_pixels_count["blue"] += 1
            pixels_count["blue"] += 1
        else:
            print("oh")
    wrongness = {
        "red": wrong_pixels_count["red"] / pixels_count["red"] if pixels_count["red"] > 0 else 0,
        "blue": wrong_pixels_count["blue"] / pixels_count["blue"] if pixels_count["blue"] > 0 else 0
    }
    return wrongness


def calculate_wrongness(svm_result, pixels):
    wrong_pixels_count = {"red": 0, "blue": 0}
    pixels_count = {"red": 0, "blue": 0}
    wrongness = {
        "red": wrong_pixels_count["red"] / pixels_count["red"] if pixels_count["red"] > 0 else 0,
        "blue": wrong_pixels_count["blue"] / pixels_count["blue"] if pixels_count["blue"] > 0 else 0
    }
    return wrongness


def plot_results(results):
    for svc_kernel in SVC_KERNELS:
        if svc_kernel == "rbf":
            for gamma in GAMMAS:
                margins = []
                wrongnesses = {"red": [], "blue": []}
                for c in C_COEFFICIENTS:
                    margin, wrongness, _ = results[svc_kernel][gamma][c]
                    margins.append(margin)
                    wrongnesses["red"].append(wrongness["red"])
                    wrongnesses["blue"].append(wrongness["blue"])
                plt.title("svm, svc_kernel: " + svc_kernel + ", gamma: " + str(gamma))
                plt.xlabel("c")
                plt.ylabel("Error rate")
                plt.title("Error rate, svc_kernel: rbf, gamma: %f" % (gamma,))
                plt.plot(C_COEFFICIENTS, wrongnesses["red"], "r")
                plt.plot(C_COEFFICIENTS, wrongnesses["blue"], "b")
                plt.show()
                plt.title("Margin, svc_kernel: " + svc_kernel)
                plt.xlabel("c")
                plt.ylabel("Margin")
                plt.plot(C_COEFFICIENTS, margins, "r")
                plt.plot(C_COEFFICIENTS, margins, "b")
                plt.show()
        else:
            margins = []
            wrongnesses = {"red": [], "blue": []}
            for c in C_COEFFICIENTS:
                margin, wrongness, _ = results[svc_kernel][c]
                margins.append(margin)
                wrongnesses["red"].append(wrongness["red"])
                wrongnesses["blue"].append(wrongness["blue"])
            plt.title("Error rate,, svc_kernel: " + svc_kernel)
            plt.xlabel("c")
            plt.ylabel("Error rate")
            plt.plot(C_COEFFICIENTS, wrongnesses["red"], "r")
            plt.plot(C_COEFFICIENTS, wrongnesses["blue"], "b")
            plt.show()
            plt.title("Margin, svc_kernel: " + svc_kernel)
            plt.xlabel("c")
            plt.ylabel("Margin")
            plt.plot(C_COEFFICIENTS, margins, "r")
            plt.plot(C_COEFFICIENTS, margins, "b")
            plt.show()


def run():
    pixels, labels, shape = load_data()
    results = {svc_kernel: {c_coefficient: {} for c_coefficient in C_COEFFICIENTS} for svc_kernel in SVC_KERNELS}
    results["rbf"] = {}
    for c in C_COEFFICIENTS:
        svm_result = run_svm(pixels, labels, "linear", c)
        a, xx, yy, margin = calculate_data(svm_result, pixels, "linear", c, labels)
        # print(pixels[0])
        wrongness = calculate_wrongness_linear(svm_result, a, xx, yy, pixels)
        results["linear"][c] = (margin, wrongness, svm_result)
        # print("=============")

    for c in C_COEFFICIENTS:
        svm_result = run_svm(pixels, labels, "poly", c)
        margin = calculate_data(svm_result, pixels, "poly", c, labels)
        # a, xx, yy, margin = show_svm_result(svm_result, pixels, "poly", labels)
        # wrongness = calculate_wrongness(svm_result, a, xx, yy, pixels)
        wrongness = calculate_wrongness(svm_result, pixels)
        results["poly"][c] = (margin, wrongness, svm_result)

    for gamma in GAMMAS:
        results["rbf"][gamma] = {}
        for c in C_COEFFICIENTS:
            svm_result = run_svm(pixels, labels, "rbf", c, gamma)
            # a, xx, yy, margin = show_svm_result(svm_result, pixels, "rbf", labels)
            # wrongness = calculate_wrongness(svm_result, a, xx, yy, pixels)
            margin = calculate_data(svm_result, pixels, "rbf", c, labels)
            wrongness = calculate_wrongness(svm_result, pixels)
            # print(svm_result.support_vectors_)
            results["rbf"][gamma][c] = (margin, wrongness, svm_result)

    plot_results(results)
    # print(results["linear"])

    for c in C_COEFFICIENTS:
        for svc_kernel in SVC_KERNELS:
            if svc_kernel == "rbf":
                for gamma, gamma_result in results[svc_kernel].items():
                    margin, wrongness, svm_result = gamma_result[c]

                    X = pixels
                    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                                         np.arange(y_min, y_max, 0.2))

                    clf = svm_result
                    # plt.subplot(2, 2, i + 1)
                    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
                    print(Z)

                    # Put the result into a color plot
                    # Z0 = Z[:, 0].reshape(xx.shape)
                    # Z1 = Z[:, 1].reshape(xx.shape)
                    # plt.contourf([[x_ if x_ > svm_result.predict([x_]) else None for x_ in x] for x in xx], [[y_ if y_ > svm_result.predict([y_]) else None for y_ in y] for y in yy], Z0, c=["#DD2222"], alpha=0.8)
                    # plt.contourf(xx, yy, Z0, c=["#FFFFFF"])
                    # plt.contourf(xx, yy, Z1, c=["#FFFFFF"])
                    Z = np.array([1 - abs(z[0] - z[1]) for z in Z]).reshape(xx.shape)
                    plt.contourf(xx, yy, Z)

                    # Plot also the training points
                    plt.scatter(X[:, 0], X[:, 1], c=["#FF0000" if clf.predict([x]) > 0 else "#0000FF" for x in X])
                    plt.xlim(xx.min(), xx.max())
                    plt.ylim(yy.min(), yy.max())
                    plt.xticks(range(x_min, x_max, 20))
                    plt.yticks(range(y_min, y_max, 20))
                    plt.title("space division, svc_kernel = %s, gamma = %f, c = %f" % (svc_kernel, gamma, c))

                    plt.show()
            else:
                margin, wrongness, svm_result = results[svc_kernel][c]
                X = pixels
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                                     np.arange(y_min, y_max, 0.2))

                clf = svm_result
                # plt.subplot(2, 2, i + 1)
                # plt.subplots_adjust(wspace=0.4, hspace=0.4)

                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
                print(Z)

                # Put the result into a color plot
                # Put the result into a color plot
                # Z0 = Z[:, 0].reshape(xx.shape)
                # Z1 = Z[:, 1].reshape(xx.shape)
                Z = np.array([1 - abs(z[0] - z[1]) for z in Z]).reshape(xx.shape)
                plt.contourf(xx, yy, Z)
                # plt.contourf(xx, yy, Z1, c=["#FFFFFF"])

                # Plot also the training points
                plt.scatter(X[:, 0], X[:, 1], c=["#FF0000" if clf.predict([x]) > 0 else "#0000FF" for x in X])
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.xticks(range(x_min, x_max, 20))
                plt.yticks(range(y_min, y_max, 20))
                plt.title("space division, svc_kernel = %s, c = %f" % (svc_kernel, c))

                plt.show()


if __name__ == "__main__":
    run()
