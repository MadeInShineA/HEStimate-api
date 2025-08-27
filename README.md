# HEStimate API
<!-- BADGES:START -->
[![CI](https://github.com/MadeInShineA/HEStimate-api/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MadeInShineA/HEStimate-api/actions/workflows/ci.yml) ![Tests](https://img.shields.io/badge/Tests-10/10%20passing-brightgreen) ![Coverage](https://img.shields.io/badge/Coverage-94.8%25-brightgreen)
<!-- BADGES:END -->

FastAPI API for **face verification and comparison** using DeepFace.

## Prerequisites

- [Python 3.11+](https://www.python.org/)
---

## Installation

- Create a venv and activate it following the [python doc](https://docs.python.org/3/library/venv.html)
- Install the requirements

```
pip install -r requirements.txt
```

## Run the API locally
```
fastapi dev main.py 
```

The API will be available at:

```
http://127.0.0.1:8000/
```

## Documentation
FastAPI automatically generates an interactive documentation at:
```
http://127.0.0.1:8000/docs
```

## Unit Tests

Run the tests:

```
pytest -q
```

## Coverage

To measure the test coverage of the project:

```
coverage run -m pytest
coverage report -m
```

You can also generate an HTML report for easier navigation:

```
coverage html
```
