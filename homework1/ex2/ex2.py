import random
import json
import math

RADIUS = 1
SIDE = 2 * RADIUS
DIMENSIONS = 3
ITERATIONS = 4


TEST_CASES = {
    10,
    20
}


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


def calculate_point_distance(point0, point1):
    return math.sqrt(sum(map(lambda a: (a[0] - a[1]) ** 2, zip(point0, point1))))


def test_case_standard_deviation(test_case):
    standard_deviation = []
    num_of_distances = len(test_case) * (len(test_case) - 1) / 2
    mean_distance = test_case_mean_distance(test_case)
    for i in range(len(test_case)):
        for j in range(i + 1, len(test_case)):
            point_distance = calculate_point_distance(test_case[i], test_case[j])
            standard_deviation.append((point_distance - mean_distance) ** 2)
    standard_deviation = math.sqrt(sum(standard_deviation) / (num_of_distances - 1))
    return standard_deviation


def test_case_mean_distance(test_case):
    mean_distance = []
    for i in range(len(test_case)):
        for j in range(i + 1, len(test_case)):
            mean_distance.append(calculate_point_distance(test_case[i], test_case[j]))
    return sum(mean_distance) / len(mean_distance) if len(mean_distance) > 0 else 0


def calculate_standard_deviation(points):
    dimension_data = {}
    for dimension in range(1, DIMENSIONS + 1):
        iteration_data = []
        for i in range(ITERATIONS):
            test_case_data = []
            for test_case in points[i][dimension]:
                test_case_data.append(test_case_standard_deviation(test_case))
            dimension_data[dimension] = test_case_data
        iteration_data.append(dimension_data)
    return dimension_data


def calculate_mean_distance(points):
    dimension_data = {}
    for dimension in range(1, DIMENSIONS + 1):
        iteration_data = []
        for i in range(ITERATIONS):
            test_case_data = []
            for test_case in points[i][dimension]:
                test_case_data.append(test_case_mean_distance(test_case))
            iteration_data.append(test_case_data)
        dimension_data[dimension] = iteration_data
    return dimension_data


def plot_data_to_file(data):
    pass


def run():
    dimension_data = generate_points(DIMENSIONS)
    print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    mean_distance = calculate_mean_distance(dimension_data)
    standard_deviation = calculate_standard_deviation(dimension_data)

    print("MD")
    print(json.dumps(mean_distance, indent=2))
    print("SD")
    print(json.dumps(standard_deviation, indent=2))
    # ,print(list(zip(*[mean_distance[i][1] for i in range(ITERATIONS)])))
    print(list(map(lambda l: sum(l) / len(l), list(zip(*[mean_distance[i][1] for i in range(ITERATIONS)])))))
    mean_mean_distance = {dimension: list(map(lambda l: sum(l) / len(l), list(zip(*[mean_distance[i][dimension] for i in range(ITERATIONS)])))) for dimension in range(1, DIMENSIONS + 1)}
    print(json.dumps(mean_mean_distance, indent=2))
    mean_standard_deviation = {dimension: list(map(lambda l: sum(l) / len(l), list(zip(*[standard_deviation[i][dimension] for i in range(ITERATIONS)])))) for dimension in range(1, DIMENSIONS + 1)}
    print(json.dumps(mean_standard_deviation, indent=2))
    # print(json.dumps([{j: mean_distance[i][j] - standard_deviation[i][j] for j in range(1, DIMENSIONS + 1)} for i in range(len(mean_distance))], indent=2))

    # print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    # percentage_hyperball = map(lambda dimension: sum(dimension) / len(dimension), selected_points.values())
    # print(json.dumps(percentage_hyperball, indent=2))
    plot_data_to_file(dimension_data)
    # print(json.dumps(dimension_data, indent=2))


if __name__ == "__main__":
    run()
