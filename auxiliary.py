import json
import re


def get(dataset, slug_name):
    for instance in dataset:
        if instance['slug_name'] == slug_name:
            return instance


def solution(instance, idx):
    return sorted(
        instance['solutions'],
        key=lambda s: int(s['runtime'][:-2])
    )[idx]['solution']


def extract_code(dataset, outputs):
    code = re.compile(r'```python\s+(.*?)```', re.DOTALL)
    methods = dict()
    for instance in dataset:
        slug_name = instance['slug_name']
        methods[slug_name] = instance['entry_point']

    codes = dict()
    for key, val in outputs.items():
        snippets = [
            snippet
            for snippet in re.findall(code, val)
            if 'class Solution' in snippet and methods[key] in snippet
        ]
        codes[key] = snippets[-1] if snippets else ''
 
    return codes


with open('mercury.json', 'r') as file:
    mercury = json.load(file)
