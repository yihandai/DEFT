"""Utils for evaluation"""

import numpy as np


def cal_dtw(shortest_distances, prediction, reference, success=None, threshold=3.0):
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction) + 1):
        for j in range(1, len(reference) + 1):
            best_previous_cost = min(
                dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1]
            )
            cost = shortest_distances[prediction[i - 1]][reference[j - 1]]
            dtw_matrix[i][j] = cost + best_previous_cost

    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw / (threshold * len(reference)))
    if success is None:
        success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
    sdtw = success * ndtw

    return {"DTW": dtw, "nDTW": ndtw, "SDTW": sdtw}


def cal_cls(shortest_distances, prediction, reference, threshold=3.0):
    def length(nodes):
        if len(nodes) < 2:
            return 0.0
        return np.sum([shortest_distances[a][b] for a, b in zip(nodes[:-1], nodes[1:])])

    # Handle edge cases
    if len(reference) == 0:
        return 0.0  # No reference path, return 0

    if len(prediction) == 0:
        return 0.0  # No prediction path, return 0

    # Calculate coverage: for each reference node, find minimum distance to any prediction node
    coverage_values = []
    for u in reference:
        min_distances = [shortest_distances[u][v] for v in prediction]
        if len(min_distances) == 0:
            coverage_values.append(0.0)
        else:
            min_dist = np.min(min_distances)
            coverage_values.append(np.exp(-min_dist / threshold))

    if len(coverage_values) == 0:
        coverage = 0.0
    else:
        coverage = np.mean(coverage_values)

    # Handle NaN in coverage (e.g., if shortest_distances contains NaN)
    if np.isnan(coverage):
        return 0.0

    ref_length = length(reference)
    pred_length = length(prediction)
    expected = coverage * ref_length

    # Handle division by zero: if expected and pred_length are both 0, score should be 1.0
    denominator = expected + np.abs(expected - pred_length)
    if denominator == 0:
        score = 1.0
    else:
        score = expected / denominator

    # Handle NaN in score
    if np.isnan(score):
        return 0.0

    result = coverage * score

    # Final check for NaN
    if np.isnan(result):
        return 0.0

    return result
