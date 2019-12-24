import random
import json
from decimal import *
import math
import matplotlib.pyplot as plt
import statistics

RADIUS = 1
SIDE = 2 * RADIUS
DIMENSIONS = 5
ITERATIONS = 10


TEST_CASES = {
    10,
    20,
    50,
    100,
    200,
    500,
    1000
}


def point_in_hyperball(point):
    return sum(map(lambda coordinate: coordinate ** 2, point)) < RADIUS ** 2


def generate_points(dimensions):
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_points = []
            for test_case in TEST_CASES:
                points = [
                    [random.uniform(-1.0, 1.0) for _ in range(dimension)] for _ in range(test_case)
                ]
                test_case_points.append(points)
            dimension_data[dimension] = test_case_points
        iteration_data.append(dimension_data)
    return iteration_data


def select_points_inside_hyperball(data):
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_hyperball_ratios = []
            for test_case in data[i][dimension]:
                points_in_hyperball = 0
                for point in test_case:
                    if point_in_hyperball(point):
                        points_in_hyperball += 1
                hyperball_ratio = points_in_hyperball / len(test_case)
                test_case_hyperball_ratios.append(hyperball_ratio)
            dimension_data[dimension] = test_case_hyperball_ratios
        iteration_data.append(dimension_data)
    return iteration_data


def iteration_mean(data):
    mean_data = {dimension: [[] for _ in range(len(TEST_CASES))] for dimension in range(1, DIMENSIONS + 1)}
    for i in range(ITERATIONS):
        for dimension in range(1, DIMENSIONS + 1):
            for j in range(len(TEST_CASES)):
                mean_data[dimension][j].append(data[i][dimension][j])
    mean = {dimension: [] for dimension in range(1, DIMENSIONS + 1)}
    for dimension in range(1, DIMENSIONS + 1):
        for test_case in range(len(TEST_CASES)):
            mean[dimension].append(sum(mean_data[dimension][test_case]) / len(mean_data[dimension][test_case]))
    return mean


def test_case_standard_deviation(test_case):
    standard_deviation = []
    print(test_case)
    mean = sum(test_case) / len(test_case)
    for i in range(len(test_case)):
        standard_deviation.append((test_case[i] - mean) ** 2)
    standard_deviation = math.sqrt(sum(standard_deviation) / (len(test_case) - 1))
    return standard_deviation


def iteration_standard_deviation(iteration_data):
    return list(map(lambda test_case: statistics.stdev(test_case), iteration_data))


def calculate_standard_deviation(data):
    dimension_data = {}
    for dimension in range(1, DIMENSIONS + 1):
        iteration_data = []
        for i in range(ITERATIONS):
            iteration_data.append(data[i][dimension])
        dimension_data[dimension] = iteration_standard_deviation(iteration_data)
    # dimension_data = {dimension: [] for dimension in range(1, DIMENSIONS + 1)}
    # iteration_data = []
    # for i in range(ITERATIONS):
    #     dimension_data = {}
    #     for dimension in range(1, DIMENSIONS + 1):
    #         test_case_data = []
    #         for test_case in points[i][dimension]:
    #             test_case_data.append(test_case_standard_deviation(test_case))
    #         dimension_data[dimension] = test_case_data
    #     iteration_data.append(dimension_data)
    return dimension_data


def plot_data_to_file(data):
    mean = iteration_mean(data)
    print(json.dumps(mean, indent=2))
    standard_deviation = calculate_standard_deviation(data)
    print(json.dumps(standard_deviation, indent=2))


def run():
    getcontext().prec = 10
    iteration_data = generate_points(DIMENSIONS)
    selected_points = select_points_inside_hyperball(iteration_data)
    print(json.dumps(selected_points, indent=2))
    plot_data_to_file(selected_points)
    # print(json.dumps(dimension_data, indent=2))
    # percentage_hyperball = map(lambda dimension: sum(dimension) / len(dimension), selected_points.values())
    # print(json.dumps(percentage_hyperball, indent=2))\


if __name__ == "__main__":
    run()
