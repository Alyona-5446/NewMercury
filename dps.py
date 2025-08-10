import math
from typing import Any, Callable

import numpy as np


SlugName = str
MetricName = str
MetricValue = float
Results = dict[SlugName, dict[MetricName, MetricValue | str]]
Distributions = dict[SlugName, dict[MetricName, list[list[MetricValue]]]]
DistributionFilter = Callable[[Distributions, str, ...], Distributions]


def filter_by_compute_cost(
    dists: Distributions,
    metric_name: str,
    min_metric_value: float,
    **kwargs: Any,
) -> Distributions:
    return {
        slug_name: solutions
        for slug_name, solutions in dists.items()
        if min(np.mean(solution) for solution in solutions[metric_name]) >= min_metric_value
    }


def filter_by_cv(
    dists: Distributions,
    metric_name: str,
    cv_percentile: int,
    max_cv: float,
    **kwargs: Any,
) -> Distributions:
    return {
        slug_name: solutions
        for slug_name, solutions in dists.items()
        if np.percentile(
            [get_cv(solution) for solution in solutions[metric_name]],
            cv_percentile,
        ) <= max_cv
    }


def filter_by_clusters(
    dists: Distributions,
    metric_name: str,
    min_clusters: int,
    cluster_ratio_bias: float,
    cluster_ratio_weight: float,
    **kwargs: Any,
) -> Distributions:
    return {
        slug_name: solutions
        for slug_name, solutions in dists.items()
        if len(get_clusters(
            solutions[metric_name],
            cluster_ratio_bias,
            cluster_ratio_weight,
        )) >= min_clusters
    }


def get_cv(values: list[float]) -> float:
    return np.std(values) / np.mean(values)


def get_cluster_threshold(metric_value: float, cluster_ratio_bias: float, cluster_ratio_weight: float) -> float:
    return cluster_ratio_bias + math.sqrt(cluster_ratio_weight / metric_value)


def get_clusters(
    metrics: list[float],
    cluster_ratio_bias: float = 0.2,
    cluster_ratio_weight: float = 0,
    **kwargs: Any,
) -> list[list[float]]:
    relative_distance = -np.diff(metrics) / metrics[:-1]

    indices = []
    for idx, rd in enumerate(relative_distance):
        if rd > get_cluster_threshold(metrics[idx], cluster_ratio_bias, cluster_ratio_weight):
            indices.append(idx + 1)

    return np.split(metrics, indices)


def calc_dps(
    results: Results,
    dists: Distributions,
    metric_name: str,
    filters: list[DistributionFilter] | None = None,
    **kwargs: Any,
) -> tuple[float, float, int]:
    if filters is not None:
        for filter in filters:
            dists = filter(dists, metric_name, **kwargs)

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
        clusters = get_clusters(mean_metrics, **kwargs)

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
