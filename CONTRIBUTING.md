# Contributing

## Dev Setup

```bash
git clone https://github.com/alexmar07/llmkit.git
cd llmkit
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the package in editable mode together with `pytest`, `pytest-asyncio`,
`pytest-cov`, `respx`, `ruff`, and `mypy`.

To also build or preview the documentation:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Running Checks

All three checks must pass before opening a pull request:

```bash
# Lint and format check
ruff check src/ tests/

# Static type checking
mypy src/llmkit/

# Tests with coverage report
pytest tests/ -v --cov=src/llmkit --cov-report=term-missing
```

The CI pipeline runs these checks on Python 3.12 and 3.13.

## Code Style

- **Type hints** are required on every function and method signature, including
  `self`-less module-level functions.
- **Ruff** enforces PEP 8, import ordering, and a set of quality rules (see
  `[tool.ruff.lint]` in `pyproject.toml`). Run `ruff check --fix` to auto-fix
  safe violations.
- **Mypy strict mode** is enabled. `# type: ignore` comments require a justifying
  comment and should be rare.
- **Docstrings** use Google style and are required on all public classes and methods.
- Lines are limited to **100 characters**.

## Adding a New Provider

1. Create `src/llmkit/providers/<name>.py` with a class `<Name>Provider`.
   The class must satisfy the `Provider` protocol (see
   [`src/llmkit/provider.py`](src/llmkit/provider.py)):
   - `name` property returning the lowercase provider string
   - `async def chat(...)` returning `ChatResponse`
   - `async def stream(...)` yielding `StreamChunk` objects

2. Export the class from `src/llmkit/providers/__init__.py`.

3. Add the provider to `_PROVIDER_MAP` in `src/llmkit/client.py`:
   ```python
   _PROVIDER_MAP: dict[str, type[Any]] = {
       "openai": OpenAIProvider,
       "anthropic": AnthropicProvider,
       "ollama": OllamaProvider,
       "<name>": <Name>Provider,   # add here
   }
   ```

4. Write tests in `tests/test_<name>_provider.py`. Use `respx` to mock HTTP
   calls — no real API credentials should appear in tests.

5. Document the provider in `docs/getting-started.md` and update the provider
   support table in `README.md`.

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your changes and ensure all checks pass (lint, type check, tests).
3. Add or update tests so coverage stays above 80 %.
4. Update documentation if you added or changed public API.
5. Open a pull request against `main` with a clear description of what changes
   and why.

Pull requests are reviewed for correctness, type safety, test coverage, and
consistency with the existing code style.
