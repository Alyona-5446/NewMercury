import json


def get(dataset, slug_name):
    for instance in dataset:
        if instance['slug_name'] == slug_name:
            return instance


def solution(instance, idx):
    return sorted(
        instance['solutions'],
        key=lambda s: int(s['runtime'][:-2])
    )[idx]['solution']


with open('mercury.json', 'r') as file:
    mercury = json.load(file)
