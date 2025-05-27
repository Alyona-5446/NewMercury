import json

from collections import defaultdict
from tqdm import tqdm

from sandbox import Sandbox


def run_solution(instance, solution, run_cnt):
    runtime = 0
    for run in range(run_cnt):
        sample = {
            'solution': solution,
            'convert_offline': instance['convert_offline'],
            'evaluate_offline': instance['evaluate_offline'],
            'entry_point': instance['entry_point'],
            'test_cases': json.loads(instance['test_cases']),
            'timeout': 60
        }
        result = Sandbox.run_sample(sample)
        if result['result'] == 'passed':
            runtime += result['runtime']
        else:
            return result
    return {'result': 'passed', 'runtime': runtime / run_cnt}


def get_runtime_distributions(run_cnt):
    with open('mercury.json', 'r') as dataset_file:
        dataset = json.load(dataset_file)

    dists = dict()
    failed = defaultdict(list)
    for instance in tqdm(dataset):
        slug_name = instance['slug_name']
        runtimes = dict()
        for idx, solution in enumerate(instance['solutions']):
            result = run_solution(instance, solution['solution'], run_cnt)
            if result['result'] == 'passed':
                runtimes[idx] = result['runtime']
            else:
                failed[slug_name].append((idx, result['result']))
        dists[slug_name] = runtimes

    print(f'Failed: {len(failed)}')
    with open('failed_solutions.json', 'w') as failed_file:
        failed_file.write(json.dumps(failed))

    with open(f'runtime_distributions.json', 'w') as dist_file:
        dist_file.write(json.dumps(dists))


def pass_at_1(results):
    passed = 0
    for result in results.values():
        if result['result'] == 'passed':
            passed += 1
    return passed / len(results)


def beyond(results):
    with open('final_dists.json', 'r') as dist_file:
        dists = json.load(dist_file)
 
    beyond = 0
    for slug_name, result in results.items():
        if result['result'] == 'passed':
            min_time = dists[slug_name][0]
            max_time = dists[slug_name][-1]
            runtime = result['runtime']
            runtime = max(runtime, min_time)
            runtime = min(runtime, max_time)
            beyond += (max_time - runtime) / (max_time - min_time)
    return beyond / len(results)


def get_results(completion_path, transform, run_cnt):
    with open('mercury.json', 'r') as dataset_file:
        dataset = json.load(dataset_file)

    with open(completion_path, 'r') as completion_file:
        completions = json.load(completion_file)
 
    results = dict()
    for instance in tqdm(dataset):
        slug_name = instance['slug_name']
        completion = transform(completions[slug_name])
        results[slug_name] = run_solution(instance, completion, run_cnt)

    with open('results.json', 'w') as result_file:
        result_file.write(json.dumps(results))
 
    easy = dict()
    medium = dict()
    hard = dict()
    for instance in dataset:
        slug_name = instance['slug_name']
        if instance['difficulty'] == 'Easy':
            easy[slug_name] = results[slug_name]
        elif instance['difficulty'] == 'Medium':
            medium[slug_name] = results[slug_name]
        else:
            hard[slug_name] = results[slug_name]

    subsets = {
        'Easy': easy,
        'Medium': medium,
        'Hard': hard,
        'Total': results
    }
    print(f'Results for {completion_path}\n')
    for name, subset in subsets.items():
        print(f'{name} ({len(subset)} problems)')
        print(f'Pass@1: {pass_at_1(subset)}')
        print(f'Beyond: {beyond(subset)}\n')


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
