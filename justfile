# Justfile for vecalex

#--------------------------------------
# Variables
#--------------------------------------

build_dir := "build/"
dist_dir := "dist/"

# --------------------------------------
# Misc Commands
#--------------------------------------

# Show available commands
list:
    @just --list

VERSION := `grep -m1 '^version' pyproject.toml | sed -E 's/version = "(.*)"/\1/'`

# Print the current version of the project
version:
    @echo "Current version is {{VERSION}}"

#--------------------------------------
# QA Commands
#--------------------------------------

format:
    uv run --group lint ruff format .  # format code
    uv run --group lint ruff check --select I,RUF022 --fix .  # sort imports and __all__ exports

lint:
    uv run --group lint ruff format . --exit-non-zero-on-format  # check formatting
    uv run --group lint ruff check . --fix --exit-non-zero-on-fix  # check for and fix lint issues, but exit non-zero if fixes were made
    uv run --group lint ty check .  # type check code

# Run all the formatting, linting, and testing commands
qa: format lint test

# Run all the tests for all the supported Python versions
testall:
    uv run --python=3.10 --group test pytest
    uv run --python=3.11 --group test pytest
    uv run --python=3.12 --group test pytest
    uv run --python=3.13 --group test pytest

# Run all the tests, but allow for arguments to be passed
test *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --group test pytest {{ARGS}}

# Run all the tests, but on failure, drop into the debugger
pdb *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run  --group test pytest --pdb --maxfail=10 --pdbcls=IPython.terminal.debugger:TerminalPdb {{ARGS}}

# Run coverage, and build to HTML
coverage:
    uv run --group test coverage run -m pytest .
    uv run --group test coverage report -m
    uv run --group test coverage html

install-pre-commit-hook:
    echo "#!/bin/sh" > .git/hooks/pre-commit
    echo "just lint" >> .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

#--------------------------------------
# Build and Release Commands
#--------------------------------------

# Build the project, useful for checking that packaging is correct
build: clean-build-dist
    uv build

# Tag the current version in git and put to github
tag:
    echo "Tagging version v{{VERSION}}"
    git tag -a v{{VERSION}} -m "Creating version v{{VERSION}}"
    git push origin v{{VERSION}}

#--------------------------------------
# Clean Commands
#--------------------------------------

# remove all build, test, coverage and Python artifacts
clean: clean-build clean-pyc clean-test

# remove build artifacts
clean-build-dist:
    rm -rf {{ build_dir }}
    rm -rf {{ dist_dir }}

clean-build: clean-build-dist
    rm -rf .eggs/
    find . -not -path "./.venv/*" -name '*.egg-info' -exec rm -rf {} +
    find . -not -path "./.venv/*" -name '*.egg' -exec rm -f {} +

# remove Python file artifacts
clean-pyc:
    find . -name '*.pyc' -exec rm -f {} +
    find . -name '*.pyo' -exec rm -f {} +
    find . -name '*~' -exec rm -f {} +
    find . -name '__pycache__' -exec rm -rf {} +

# remove test and coverage artifacts
clean-test:
    rm -f .coverage
    rm -rf htmlcov/
    rm -rf .pytest_cache
