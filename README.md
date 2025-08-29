# NewMercury
NewMercury is an efficiency evaluation framework for LLM-generated code. It is based on [Mercury](https://github.com/Elfsong/Mercury) and [EvalPerf](https://github.com/evalplus/evalplus), two code efficiency benchmarks.

### Resource usage profiling
- runtime (`time.perf_counter`)
- CPU instruction count (`cirron.Collector`)
- peak memory usage (`tracemalloc.get_traced_memory`)

### Code quality metrics
- Pass@1 ([HumanEval](https://github.com/openai/human-eval))
- Beyond ([Mercury](https://github.com/Elfsong/Mercury))
- DPS and Normalised DPS ([EvalPerf](https://github.com/evalplus/evalplus))

### NewMercury VS Mercury
- corrected test generation, input conversion and solution evaluation functions
- implemented test generation facilities and extended test suites
- enhanced sandbox with additional library imports and adjusted time/recursion limits
- new types of resource usage measurements and additional efficiency metrics
- metric calculation using precomputed resource usage distributions for canonical solutions

## How to use NewMercury
- **solution generation:** `Mercury.ipynb`
- **LLM evaluation:** `get_results` and `get_metrics` (`evaluate.py`)
- **test generation:** `Sandbox.generate_tests` (`gen_tests.py`)
