import random
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
from decimal import Decimal

RADIUS = 1
SIDE = 2 * RADIUS
DIMENSIONS = 15
ITERATIONS = 10


TEST_CASES = [
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000
]


def generate_points(dimensions):
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_points = {}
            for test_case in TEST_CASES:
                coordinates = [
                    [random.uniform(-1.0, 1.0) for _ in range(dimension)] for _ in range(test_case)
                ]
                test_case_points[test_case] = coordinates
            dimension_data[dimension] = test_case_points
        iteration_data.append(dimension_data)
    return iteration_data


def calculate_standard_deviation(data):
    dimension_data = {}
    for dimension in range(1, DIMENSIONS + 1):
        test_case_data = {}
        for test_case in TEST_CASES:
            iteration_data = []
            for i in range(ITERATIONS):
                iteration_data.append(data[i][dimension][test_case])
            test_case_data[test_case] = statistics.stdev(iteration_data)
        dimension_data[dimension] = test_case_data
    return dimension_data


def calculate_mean(data):
    dimension_data = {}
    for dimension in range(1, DIMENSIONS + 1):
        test_case_data = {}
        for test_case in TEST_CASES:
            iteration_data = []
            for i in range(ITERATIONS):
                iteration_data.append(data[i][dimension][test_case])
            test_case_data[test_case] = statistics.mean(iteration_data)
        dimension_data[dimension] = test_case_data
    return dimension_data


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))


def get_angle_between_points(four_points):
    first_pair = np.array([a - b for a, b in zip(*four_points[0:2])])
    second_pair = np.array([a - b for a, b in zip(*four_points[2:4])])
    return angle_between(first_pair, second_pair)


def get_angles_of_random_pairs_of_points(data):
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_angles = {}
            for test_case in TEST_CASES:
                random_four_points = random.sample(data[i][dimension][test_case], 4)
                test_case_angles[test_case] = get_angle_between_points(random_four_points)
            dimension_data[dimension] = test_case_angles
        iteration_data.append(dimension_data)
    return iteration_data


def plot_data(mean, standard_deviation):
    for dimension in range(1, DIMENSIONS + 1):
        dimension_mean = mean[dimension]
        dimension_standard_deviation = standard_deviation[dimension]
        plt.xticks(TEST_CASES)
        plt.xscale("log")
        plt.scatter(dimension_mean.keys(), dimension_mean.values())
        plt.errorbar(dimension_mean.keys(), dimension_mean.values(),
                     xerr=[0 for _ in dimension_standard_deviation.keys()], yerr=dimension_standard_deviation.values())
        axes = plt.gca()
        axes.set_ylim(0.0, np.pi)
        axes.set_xlim(TEST_CASES[0], TEST_CASES[-1] * 2)
        plt.xlabel("Number of points")
        plt.ylabel("Stddev to mean ratio")
        plt.title("Stddev to mean ratios after 20 iterations for dimension no " + str(dimension))
        plt.show()
    for i, test_case in enumerate(TEST_CASES):
        means = []
        standard_deviations = []
        for dimension in range(1, DIMENSIONS + 1):
            means.append(mean[dimension][test_case])
        for dimension in range(1, DIMENSIONS + 1):
            standard_deviations.append(standard_deviation[dimension][test_case])
        axes = plt.gca()
        axes.set_ylim(0.0, np.pi)
        axes.set_xlim(1, DIMENSIONS)
        plt.xticks(range(0, DIMENSIONS + 1))
        plt.scatter(range(1, DIMENSIONS + 1), means, cmap="rainbow")
        plt.errorbar(range(1, DIMENSIONS + 1), means,
                     xerr=[0 for _ in range(1, DIMENSIONS + 1)], yerr=standard_deviations)
        plt.xlabel("Dimensions")
        plt.ylabel("Stddev to mean ratio")
        plt.title("Stddev to mean ratios after 20 iterations for test case no " + str(i + 1) + " (" + str(test_case) + " points)")
        plt.show()


def run():
    data = generate_points(DIMENSIONS)
    # print(json.dumps(get_random_pairs_of_points(data), indent=2))
    mean_angles = get_angles_of_random_pairs_of_points(data)
    means = calculate_mean(mean_angles)
    standard_deviations = calculate_standard_deviation(mean_angles)
    # mean_angles = {dimension: list(map(lambda l: sum(l) / len(l), list(zip(*[mean_angles[i][dimension] for i in range(ITERATIONS)])))) for dimension in range(1, DIMENSIONS + 1)}
    print(json.dumps(mean_angles, indent=2))

    # print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    # percentage_hyperball = map(lambda dimension: sum(dimension) / len(dimension), selected_points.values())
    # print(json.dumps(percentage_hyperball, indent=2))
    plot_data(means, standard_deviations)


if __name__ == "__main__":
    run()
