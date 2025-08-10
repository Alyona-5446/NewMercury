import math
from typing import Any, Callable

import numpy as np


SlugName = str
MetricName = str
MetricValue = float
Results = dict[SlugName, dict[MetricName, MetricValue | str]]
Distributions = dict[SlugName, dict[MetricName, list[list[MetricValue]]]]
DistributionFilter = Callable[[Distributions, str, ...], Distributions]


def filter_by_compute_cost(dists: Distributions, metric_name: str, min_threshold: float = 0) -> Distributions:
    return {
        slug_name: solutions
        for slug_name, solutions in dists.items()
        if min(np.mean(solution) for solution in solutions[metric_name]) >= min_threshold
    }


def filter_by_cv(dists: Distributions, metric_name: str, percentile: int = 99, max_cv: float = 0.05) -> Distributions:
    return {
        slug_name: solutions
        for slug_name, solutions in dists.items()
        if np.percentile(
            [get_cv(solution) for solution in solutions[metric_name]],
            percentile,
        ) <= max_cv
    }


def filter_by_clusters(
    dists: Distributions,
    metric_name: str,
    min_clusters: int = 3,
    base_threshold: float = 0.2,
    weight: float = 0,
) -> Distributions:
    return {
        slug_name: solutions
        for slug_name, solutions in dists.items()
        if len(get_clusters(
            solutions[metric_name],
            base_threshold,
            weight,
        )) >= min_clusters
    }


def get_cv(values: list[float]) -> float:
    return np.std(values) / np.mean(values)


def get_cluster_threshold(metric_value: float, base_threshold: float, weight: float) -> float:
    return base_threshold + math.sqrt(weight / metric_value)


def get_clusters(metrics: list[float], base_threshold: float = 0.1, weight: float = 0) -> list[list[float]]:
    relative_distance = -np.diff(metrics) / metrics[:-1]

    indices = []
    for idx, rd in enumerate(relative_distance):
        if rd > get_cluster_threshold(metrics[idx], base_threshold, weight):
            indices.append(idx + 1)

    return np.split(metrics, indices)


def calc_dps(
    results: Results,
    dists: Distributions,
    metric_name: str = 'runtime',
    filters: list[DistributionFilter] | None = None,
) -> tuple[float, float, int]:
    if filters is not None:
        for filter in filters:
            dists = filter(dists)

    num_tasks = len(dists)
    if not num_tasks:
        return 0, 0, 0

    dps, dps_norm = 0, 0
    for slug_name, solutions in dists.items():
        result = results[slug_name]
        if result['status'] != 'passed':
            continue

        result_metric = result[metric_name]
        mean_metrics = sorted(
            np.mean(solution)
            for solution in solutions[metric_name]
        )[::-1]
        clusters = get_clusters(mean_metrics)

        ratio = 0
        for idx, cluster in enumerate(clusters):
            reference = mean_metrics[cluster[0]]
            if reference <= result_metric:
                dps += ratio
                dps_norm += idx / len(clusters)
                break
            ratio += len(cluster) / len(mean_metrics)
        else:
            dps += 1
            dps_norm += 1

    return dps / num_tasks, dps_norm / num_tasks, num_tasks
