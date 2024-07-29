# conftest.py
import subprocess
import pytest

def pytest_configure(config):
    try:
        # Run the `git lfs pull` command
        print('git lfs pull ...')
        result = subprocess.run(["git", "lfs", "pull"], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running git lfs pull: {e.stderr}")
        pytest.exit("Exiting tests due to git lfs pull failure")
