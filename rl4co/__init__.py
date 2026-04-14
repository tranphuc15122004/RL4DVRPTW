from importlib.metadata import PackageNotFoundError, version as get_version

# The package version is obtained from the pyproject.toml file
try:
	__version__ = get_version(__package__)
except PackageNotFoundError:
	__version__ = "0.0.0+local"
