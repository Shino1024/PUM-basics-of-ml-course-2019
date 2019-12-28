import random
import json
from decimal import *
import math
import matplotlib.pyplot as plt
import statistics
from decimal import Decimal

RADIUS = 1
SIDE = 2 * RADIUS
DIMENSIONS = 15
ITERATIONS = 20


TEST_CASES = [
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000
]


def point_in_hyperball(point):
    return sum(map(lambda coordinate: coordinate ** 2, point)) <= RADIUS ** 2


def generate_points():
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_points = {}
            for test_case in TEST_CASES:
                coordinates = [
                    [Decimal(random.uniform(-1.0, 1.0)) for _ in range(dimension)] for _ in range(test_case)
                ]
                test_case_points[test_case] = coordinates
            dimension_data[dimension] = test_case_points
        iteration_data.append(dimension_data)
    return iteration_data


def select_points_inside_hyperball(data):
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_hyperball_ratios = {}
            for test_case in  TEST_CASES:
                points_in_hyperball = 0
                for point in data[i][dimension][test_case]:
                    if point_in_hyperball(point):
                        points_in_hyperball += 1
                hyperball_ratio = points_in_hyperball / test_case
                test_case_hyperball_ratios[test_case] = hyperball_ratio
            dimension_data[dimension] = test_case_hyperball_ratios
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
        axes.set_ylim(0.0, 1.0)
        axes.set_xlim(TEST_CASES[0], TEST_CASES[-1] * 2)
        plt.xlabel("Number of points")
        plt.ylabel("Hyperball ratio")
        plt.title("Hyperball ratios after 20 iterations for dimension no " + str(dimension))
        plt.show()
    for i, test_case in enumerate(TEST_CASES):
        means = []
        standard_deviations = []
        for dimension in range(1, DIMENSIONS + 1):
            means.append(mean[dimension][test_case])
        for dimension in range(1, DIMENSIONS + 1):
            standard_deviations.append(standard_deviation[dimension][test_case])
        axes = plt.gca()
        axes.set_ylim(0.0, 1.0)
        axes.set_xlim(1, 15)
        plt.xticks(range(1, DIMENSIONS + 1))
        plt.scatter(range(1, DIMENSIONS + 1), means, cmap="rainbow")
        plt.errorbar(range(1, DIMENSIONS + 1), means,
                     xerr=[0 for _ in range(1, DIMENSIONS + 1)], yerr=standard_deviations)
        plt.xlabel("Dimensions")
        plt.ylabel("Hyperball ratio")
        plt.title("Hyperball ratios after 20 iterations for test case no " + str(i + 1) + " (" + str(test_case) + " points)")
        plt.show()


def run():
    getcontext().prec = 10
    iteration_data = generate_points()
    selected_points = select_points_inside_hyperball(iteration_data)
    # print(json.dumps(selected_points, indent=2))
    standard_deviation = calculate_standard_deviation(selected_points)
    mean = calculate_mean(selected_points)
    plot_data(mean, standard_deviation)
    print(json.dumps(standard_deviation, indent=2))
    print(json.dumps(mean, indent=2))

    # plot_data_to_file(selected_points)
    # print(json.dumps(dimension_data, indent=2))
    # percentage_hyperball = map(lambda dimension: sum(dimension) / len(dimension), selected_points.values())
    # print(json.dumps(percentage_hyperball, indent=2))\


if __name__ == "__main__":
    run()
