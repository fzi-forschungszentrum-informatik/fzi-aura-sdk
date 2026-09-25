# Releasing the SDK

Executed notebooks belong to the docs and source distribution but not to the
wheel.

## Release checklist

1. Update the version in `pyproject.toml` and `src/fzi_aura/__init__.py`.
2. Run the notebooks and commit the updated outputs.
3. Build and check the distributions:

   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check dist/*
   python scripts/check_package_artifacts.py dist
   ```

4. Commit the release changes, then tag the new version (e.g. `v1.0.1`),
   push, and create a GitHub release for the tag. The release workflow builds from the tag and publishes to PyPI.
