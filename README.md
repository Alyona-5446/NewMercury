# NewMercury
NewMercury is an efficiency evaluation framework for LLM-generated code. It is based on [Mercury](https://github.com/Elfsong/Mercury), a competitive programming benchmark, with the following improvements:
- test generation facilities and extended test suites
- corrected test generation, input conversion and solution evaluation functions
- enhanced sandbox with additional library imports and adjusted time/recursion limits

## How to use NewMercury
- **Solution generation:** `Mercury.ipynb`
- **LLM evaluation:** `get_results` function from `evaluate.py`
- **Test generation:** `Sandbox.generate_tests` method from `gen_tests.py`
