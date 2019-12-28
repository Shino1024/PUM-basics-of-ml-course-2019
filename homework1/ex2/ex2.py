import random
import json
import math
from decimal import Decimal
import statistics

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
    1000
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
    iteration_data = {}
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_data = {}
            for test_case in TEST_CASES:
                test_case_data[test_case] = data[i][dimension][test_case]
            test_case_data[test_case] = test_case_standard_deviation_distance(iteration_data)
            dimension_data[dimension] = test_case_data
    return dimension_data


def calculate_mean_distance(data):
    dimension_data = {}
    for dimension in range(1, DIMENSIONS + 1):
        test_case_data = {}
        for test_case in TEST_CASES:
            iteration_data = []
            for i in range(ITERATIONS):
                print(data[i][dimension][test_case])
                iteration_data.append(data[i][dimension][test_case][0])
            test_case_data[test_case] = test_case_mean_distance(iteration_data)
        dimension_data[dimension] = test_case_data
    return dimension_data


def plot_data_to_file(data):
    pass


def run():
    data = generate_points()
    # print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    mean_distance = calculate_mean_distance(data)
    standard_deviation = calculate_standard_deviation_distance(data)

    print("MD")
    print(json.dumps(mean_distance, indent=2))
    print("SD")
    print(json.dumps(standard_deviation, indent=2))
    # ,print(list(zip(*[mean_distance[i][1] for i in range(ITERATIONS)])))
    # print(list(map(lambda l: sum(l) / len(l), list(zip(*[mean_distance[i][1] for i in range(ITERATIONS)])))))
    mean_mean_distance = {dimension: list(map(lambda l: sum(l) / len(l), list(zip(*[mean_distance[i][dimension] for i in range(ITERATIONS)])))) for dimension in range(1, DIMENSIONS + 1)}
    # print(json.dumps(mean_mean_distance, indent=2))
    mean_standard_deviation = {dimension: list(map(lambda l: sum(l) / len(l), list(zip(*[standard_deviation[i][dimension] for i in range(ITERATIONS)])))) for dimension in range(1, DIMENSIONS + 1)}
    # print(json.dumps(mean_standard_deviation, indent=2))
    # print(json.dumps([{j: mean_distance[i][j] - standard_deviation[i][j] for j in range(1, DIMENSIONS + 1)} for i in range(len(mean_distance))], indent=2))

    # print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    # percentage_hyperball = map(lambda dimension: sum(dimension) / len(dimension), selected_points.values())
    # print(json.dumps(percentage_hyperball, indent=2))
    plot_data_to_file(dimension_data)
    # print(json.dumps(dimension_data, indent=2))


if __name__ == "__main__":
    run()
