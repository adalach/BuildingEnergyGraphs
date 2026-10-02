"""
beg.validation - data-quality checks for generated graph datasets.

The package validates the whole chain that a configuration goes through:

    config JSON  ->  graph DataFrames  ->  NetworkX graphs  ->  PyG datasets

Every check is a plain function that returns ``list[CheckResult]`` so the same
logic can be used from pytest (``tests/graph_quality``), from the CLI report
(``scripts/run_graph_quality_checks.py``) or interactively in a notebook.

Sub-modules
-----------
core            CheckResult / Report containers and helpers
spec            ConfigSpec (expected node types, edge triples, weather variant)
                and the domain rules for feature columns (ranges, booleans,
                one-hot groups, physical consistency)
config_checks   static checks on the config JSON + SQL transform files
dataframe_checks checks on the pipeline ``out`` dictionary (node/edge tables)
nx_checks       checks on built NetworkX graphs
pyg_checks      checks on PyG ``Data``/``HeteroData`` files and
                ``training_ready.pt`` packages (torch optional)
label_checks    checks on ``labels.csv`` versus the dataset
cross_config_checks consistency of the same building across configurations
loaders         helpers to locate and load datasets on disk
"""

from beg.validation.core import CheckResult, Report, Severity, Status  # noqa: F401
from beg.validation.spec import ConfigSpec, load_config_spec  # noqa: F401
