import random
import json
import math
from decimal import Decimal
import statistics
import matplotlib.pyplot as plt

RADIUS = 1
SIDE = 2 * RADIUS
DIMENSIONS = 10
ITERATIONS = 5


TEST_CASES = [
    10,
    20,
    50,
    100,
    200,
    500
]


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


def calculate_point_distance(point0, point1):
    return math.sqrt(sum(map(lambda a: (a[0] - a[1]) ** 2, zip(point0, point1))))


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


def test_case_standard_deviation_distance(test_case):
    distances = []
    for i in range(len(test_case)):
        for j in range(i + 1, len(test_case)):
            distances.append(calculate_point_distance(test_case[i], test_case[j]))
    return statistics.stdev(distances)


def test_case_mean_distance(test_case):
    distances = []
    for i in range(len(test_case)):
        for j in range(i + 1, len(test_case)):
            distances.append(calculate_point_distance(test_case[i], test_case[j]))
    return statistics.mean(distances)


def calculate_standard_deviation_distance(data):
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_data = {}
            for test_case in TEST_CASES:
                test_case_data[test_case] = test_case_standard_deviation_distance(data[i][dimension][test_case])
            dimension_data[dimension] = test_case_data
        iteration_data.append(dimension_data)
    return iteration_data


def calculate_mean_distance(data):
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_data = {}
            for test_case in TEST_CASES:
                test_case_data[test_case] = test_case_mean_distance(data[i][dimension][test_case])
            dimension_data[dimension] = test_case_data
        iteration_data.append(dimension_data)
    return iteration_data


def calculate_ratios(mean_distance, standard_deviation_distance):
    ratios = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_data = {}
            for test_case in TEST_CASES:
                test_case_data[test_case] =\
                    standard_deviation_distance[i][dimension][test_case] / mean_distance[i][dimension][test_case]
            dimension_data[dimension] = test_case_data
        ratios.append(dimension_data)
    return ratios


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
        axes.set_ylim(0.0, 1.0)
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
    data = generate_points()
    # print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    mean_distance = calculate_mean_distance(data)
    standard_deviation = calculate_standard_deviation_distance(data)

    ratios = calculate_ratios(mean_distance, standard_deviation)

    # print("MD")
    # print(json.dumps(mean_distance, indent=2))
    # print("SD")
    # print(json.dumps(standard_deviation, indent=2))
    means = calculate_mean(ratios)
    standard_deviations = calculate_standard_deviation(ratios)
    print(json.dumps(means, indent=2))
    print(json.dumps(standard_deviations, indent=2))
    # mean_mean_distance = calculate_mean(mean_distance)
    # print(json.dumps(mean_mean_distance, indent=2))
    # mean_standard_deviation_distance = calculate_mean(standard_deviation)
    # print(json.dumps(mean_standard_deviation_distance, indent=2))
    #
    # standard_deviation_mean_distance = calculate_standard_deviation(mean_distance)
    # print(json.dumps(standard_deviation_mean_distance, indent=2))
    # standard_deviation_standard_deviation_distance = calculate_standard_deviation(standard_deviation)
    # print(json.dumps(standard_deviation_standard_deviation_distance, indent=2))
    #
    # ratios = calculate_ratios(mean_mean_distance, mean_standard_deviation_distance)
    # print(json.dumps(ratios, indent=2))
    # ,print(list(zip(*[mean_distance[i][1] for i in range(ITERATIONS)])))
    # print(list(map(lambda l: sum(l) / len(l), list(zip(*[mean_distance[i][1] for i in range(ITERATIONS)])))))
    # mean_mean_distance = {dimension: list(map(lambda l: sum(l) / len(l), list(zip(*[mean_distance[i][dimension] for i in range(ITERATIONS)])))) for dimension in range(1, DIMENSIONS + 1)}
    # print(json.dumps(mean_mean_distance, indent=2))
    # mean_standard_deviation = {dimension: list(map(lambda l: sum(l) / len(l), list(zip(*[standard_deviation[i][dimension] for i in range(ITERATIONS)])))) for dimension in range(1, DIMENSIONS + 1)}
    # print(json.dumps(mean_standard_deviation, indent=2))
    # print(json.dumps([{j: mean_distance[i][j] - standard_deviation[i][j] for j in range(1, DIMENSIONS + 1)} for i in range(len(mean_distance))], indent=2))

    # print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    # percentage_hyperball = map(lambda dimension: sum(dimension) / len(dimension), selected_points.values())
    # print(json.dumps(percentage_hyperball, indent=2))
    plot_data(means, standard_deviations)
    # print(json.dumps(dimension_data, indent=2))


if __name__ == "__main__":
    run()
