from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve

import random

def generate_dataset(
        classes_no,
        separated_well,
        foreign_island,
        different_densities,
        irregular_shape_of_one_class,
        asymetric_domain_shape
):
    result = {}
    pass


def run_classifier_on_dataset(knn, dataset):
    # knn.fit(dataset)
    pass


def print_info_to_file(result):
    pass


def run():
    knns = [
        KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean"),
        KNeighborsClassifier(n_neighbors=7, weights="uniform", metric="euclidean"),
        KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean"),
        KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="mahalanobis"),
        KNeighborsClassifier(n_neighbors=7, weights="distance", metric="mahalanobis"),
    ]
    datasets = (
        generate_dataset(5, True, True, False, True, False),
        generate_dataset(6, False, False, True, False, False),
        generate_dataset(3, False, False, False, True, True)
    )

    for knn in knns:
        for dataset in datasets:
            print_info_to_file(run_classifier_on_dataset(knn, dataset))


if __name__ == "__main__":
    run()
