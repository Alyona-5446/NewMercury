import json

from collections import defaultdict
import numpy as np
from tqdm import tqdm

from dps import calc_dps, filter_by_compute_cost, filter_by_cv, filter_by_clusters
from sandbox import Sandbox


def run_solution(instance, solution, run_cnt):
    sample = {
        'solution': solution,
        'convert_offline': instance['convert_offline'],
        'evaluate_offline': instance['evaluate_offline'],
        'entry_point': instance['entry_point'],
        'test_cases': json.loads(instance['test_cases']),
        'timeout': 60
    }
    stats = {'instr': list(), 'time': list(), 'memory': list()}

    for run in range(run_cnt):
        result = Sandbox.run_sample(sample)
        if result['status'] == 'passed':
            for stat in stats:
                stats[stat].append(result[stat])
        else:
            return result

    stats['status'] = 'passed'
    return stats


def get_distributions(run_cnt):
    with open('mercury.json', 'r') as dataset_file:
        dataset = json.load(dataset_file)

    dists = dict()
    failed = defaultdict(list)
    for instance in tqdm(dataset):
        slug_name = instance['slug_name']
        stats = {'instr': list(), 'time': list(), 'memory': list()}
        for idx, solution in enumerate(instance['solutions']):
            result = run_solution(instance, solution['solution'], run_cnt)
            if result['status'] == 'passed':
                for stat in stats:
                    stats[stat].append(result[stat])
            else:
                failed[slug_name].append((idx, result['status']))
        dists[slug_name] = stats

    with open('distributions.json', 'w') as dist_file:
        dist_file.write(json.dumps(dists))

    with open('failed_solutions.json', 'w') as failed_file:
        failed_file.write(json.dumps(failed))


def get_results(completion_path, transform, run_cnt):
    with open('mercury.json', 'r') as dataset_file:
        dataset = json.load(dataset_file)

    with open(completion_path, 'r') as completion_file:
        completions = json.load(completion_file)

    stats = ['instr', 'time', 'memory']
    results = dict()
    for instance in tqdm(dataset):
        slug_name = instance['slug_name']
        completion = transform(completions[slug_name])
        result = run_solution(instance, completion, run_cnt)
        if result['status'] == 'passed':
            for stat in stats:
                result[stat] = np.mean(result[stat])
        results[slug_name] = result

    with open('results.json', 'w') as result_file:
        result_file.write(json.dumps(results))

    with open('final_dists.json', 'r') as dist_file:
        dists = json.load(dist_file)

    subsets = {'Easy': set(), 'Medium': set(), 'Hard': set(), 'Total': set()}
    for instance in dataset:
        slug_name = instance['slug_name']
        difficulty = instance['difficulty']

        subsets['Total'].add(slug_name)
        subsets[difficulty].add(slug_name)

    filters = [filter_by_compute_cost, filter_by_cv, filter_by_clusters]
    dps_kwargs = {
        'time': {
            'min_metric_value': 0.001,  # 1ms
            'cv_percentile': 99,
            'max_cv': 0.05,
            'min_clusters': 3,
            'cluster_ratio_bias': 0.2,
            'cluster_ratio_weight': 0.001,
        },
        'instr': {
            'min_metric_value': 1_000_000,
            'cv_percentile': 99,
            'max_cv': 0.05,
            'min_clusters': 3,
            'cluster_ratio_bias': 0.2,
            'cluster_ratio_weight': 1_000_000,
        },
        'memory': {
            'min_metric_value': 16 * 1024,  # 16KB
            'cv_percentile': 99,
            'max_cv': 0.05,
            'min_clusters': 3,
            'cluster_ratio_bias': 0.2,
            'cluster_ratio_weight': 16 * 1024,
        },
    }

    metrics = defaultdict(dict)
    for name, slug_names in subsets.items():
        subset_results = {
            key: val
            for key, val in results.items()
            if key in slug_names
        }
        subset_dists = {
            key: val for key, val in dists.items()
            if key in slug_names
        }

        metrics['pass@1'][name] = calc_pass_at_1(subset_results)
        for stat in stats:
            beyond = calc_beyond(subset_results, subset_dists, stat)
            dps, dps_norm, _ = calc_dps(
                subset_results, subset_dists, stat, **dps_kwargs[stat]
            )
            dps_filt, dps_norm_filt, num_dps_tasks = calc_dps(
                subset_results, subset_dists, stat, filters, **dps_kwargs[stat]
            )
            metrics[f'beyond_{stat}'][name] = beyond
            metrics[f'dps_{stat}'][name] = dps
            metrics[f'dps_norm_{stat}'][name] = dps_norm
            metrics[f'dps_filt_{stat}'][name] = dps_filt
            metrics[f'dps_norm_filt_{stat}'][name] = dps_norm_filt
            metrics[f'dps_filt_num_tasks_{stat}'][name] = num_dps_tasks

    with open('metrics.json', 'w') as metric_file:
        metric_file.write(json.dumps(metrics))


def calc_pass_at_1(results):
    passed = 0
    for result in results.values():
        if result['status'] == 'passed':
            passed += 1
    return passed / len(results)


def calc_beyond(results, dists, metric_name):
    beyond = 0
    for slug_name, result in results.items():
        if result['status'] != 'passed':
            continue

        solutions = dists[slug_name][metric_name]
        metrics = sorted(np.mean(solution) for solution in solutions)

        min_metric = metrics[0]
        max_metric = metrics[-1]
        result_metric = min(max_metric, max(min_metric, result[metric_name]))

        beyond += (max_metric - result_metric) / (max_metric - min_metric)

    return beyond / len(results)


def split_by_newlines(completion):
    return completion.split('\n\n\n')[0].strip()


def get_main_method(completion):
    lines = completion.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith('class Solution(object):'):
            idx += 2
            break

    while idx < len(lines):
        line = lines[idx]
        spaces = 0
        while spaces < len(line) and line[spaces].isspace():
            spaces += 1
        if spaces < min(8, len(line)):
            break
        idx += 1

    return '\n'.join(lines[:idx]).strip()


get_results('phi_2_completions.json', split_by_newlines, 3)
