import random
import json
import math
import numpy as np
import matplotlib as mpl

RADIUS = 1
SIDE = 2 * RADIUS
DIMENSIONS = 20
ITERATIONS = 20


TEST_CASES = {
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    10000,
    30000
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


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_angle_between_points(four_points):
    first_pair = [a - b for a, b in zip(*four_points[0:2])]
    second_pair = [a - b for a, b in zip(*four_points[2:4])]
    return angle_between(first_pair, second_pair)


def get_angles_of_random_pairs_of_points(data):
    iteration_data = []
    for i in range(ITERATIONS):
        dimension_data = {}
        for dimension in range(1, DIMENSIONS + 1):
            test_case_angles = []
            for test_case in data[i][dimension]:
                random_four_points = random.sample(test_case, 4)
                test_case_angles.append(get_angle_between_points(random_four_points))
            dimension_data[dimension] = test_case_angles
        iteration_data.append(dimension_data)
    return iteration_data


def plot_data_to_file(data):
    pass


def run():
    data = generate_points(DIMENSIONS)
    # print(json.dumps(get_random_pairs_of_points(data), indent=2))
    mean_angles = get_angles_of_random_pairs_of_points(data)
    # mean_angles = {dimension: list(map(lambda l: sum(l) / len(l), list(zip(*[mean_angles[i][dimension] for i in range(ITERATIONS)])))) for dimension in range(1, DIMENSIONS + 1)}
    print(json.dumps(mean_angles, indent=2))

    # print(json.dumps(dimension_data, indent=2))
    # print(json.dumps(dimension_data, indent=2))
    # percentage_hyperball = map(lambda dimension: sum(dimension) / len(dimension), selected_points.values())
    # print(json.dumps(percentage_hyperball, indent=2))
    plot_data_to_file(data)


if __name__ == "__main__":
    run()
