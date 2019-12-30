from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, \
    fowlkes_mallows_score, precision_score, recall_score, f1_score, auc, average_precision_score
from PIL import Image
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import operator
from sklearn.preprocessing import label_binarize

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


def print_knn_diagnostics(knn, training_data):
    y_predicted = knn.fit(training_data["X_train"], training_data["y_train"]).predict(training_data["X_test"])
    confusion = confusion_matrix(training_data["y_test"], y_predicted)
    # recall = recall_score(training_data["y_test"], y_predicted, average="samples")
    # precision = precision_score(training_data["y_test"], y_predicted, average="samples")
    # f1 = f1_score(training_data["y_test"], y_predicted, average="samples")
    print(confusion)
    # print(recall)
    # print(precision)
    # print(f1)
    print(classification_report(training_data["y_test"], y_predicted))
    print(fowlkes_mallows_score(training_data["y_test"], y_predicted))
    # print(label_binarize(y_predicted, classes=np.unique(y_predicted)))

    y = label_binarize(training_data["y_test"], classes=np.unique(training_data["y_test"]))
    X = training_data["X_test"]
    n_classes = y.shape[1]
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    random_state = np.random.RandomState(0)
    Y = label_binarize(training_data["y_test"], classes=np.unique(training_data["y_test"]))
    n_classes = Y.shape[1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5, random_state=random_state)

    classifier = OneVsRestClassifier(svm.LinearSVC())
    classifier.fit(X_train, Y_train)
    y_score = classifier.decision_function(X_test)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))
    plt.show()
    # y_scores = knn.predict_proba(training_data["X_test"])
    # print(roc_curve(training_data["y_test"], y_scores[:, 1]))
    # for i in range(y_predicted.shape[1]):
    #     precision_recall_curve(y[], y_predicted)
    #     roc_curve(training_data["y_test"], y_predicted)


def run():
    knns = [
        KNeighborsClassifier(algorithm="brute", n_neighbors=1, weights="uniform", metric="mahalanobis", n_jobs=1000),
        KNeighborsClassifier(algorithm="brute", n_neighbors=7, weights="distance", metric="mahalanobis", n_jobs=1000),
        KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean", n_jobs=1000),
        KNeighborsClassifier(n_neighbors=7, weights="uniform", metric="euclidean", n_jobs=1000),
        KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean", n_jobs=1000),
    ]
    datasets = get_datasets()
    # print(datasets)

    percentage_results = {filename: [] for filename in FILENAMES}
    efficiency_differences = {filename: 0 for filename in FILENAMES}
    training_data = {}
    for filename in FILENAMES:
        X_train, X_test, y_train, y_test = train_test_split(datasets[filename]["model"], datasets[filename]["labels"],
                                                            test_size=0.2, random_state=0)
        training_data[filename] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        for knn in knns:
            if knn.metric == "mahalanobis":
                knn.metric_params = {"V": np.cov(X_train)}
            knn.fit(X_train, y_train)
            y_predicted = knn.predict(X_test)
            unique, counts = np.unique(y_predicted == y_test, return_counts=True)
            result = dict(zip(unique, counts))
            percentage = result[True] / sum(result.values())
            percentage_results[filename].append(percentage)
        efficiency_differences[filename] = max(percentage_results[filename]) - min(percentage_results[filename])
    print(json.dumps(percentage_results, indent=2))
    print(json.dumps(efficiency_differences, indent=2))
    max_efficiency_difference_filename = max(efficiency_differences.items(), key=operator.itemgetter(1))[0]
    print("Max efficiency difference for file " + max_efficiency_difference_filename + ".")
    best_classifier = knns[percentage_results[max_efficiency_difference_filename] \
        .index(max(percentage_results[max_efficiency_difference_filename]))]
    worst_classifier = knns[percentage_results[max_efficiency_difference_filename] \
        .index(min(percentage_results[max_efficiency_difference_filename]))]

    print_knn_diagnostics(best_classifier, training_data[max_efficiency_difference_filename])
    print_knn_diagnostics(worst_classifier, training_data[max_efficiency_difference_filename])

    h = 0.1
    for filename in FILENAMES:
        for knn in knns:
            if knn.metric == "mahalanobis":
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
