"""Building Energy Graphs (beg).

Layered pipeline for the paper's building-energy graph benchmark:

    epJSON + EPW  ->  PostgreSQL `ep` schema        (beg.ingest)
                  ->  PostgreSQL `ep_transformed`   (beg.pipeline + beg.sql)
                  ->  graph DataFrames / PyG graphs (beg.build)
                  ->  training-ready datasets       (beg.datasets)
                  ->  trained runs                  (beg.train)
                  ->  figures and tables            (beg.report)

The database is addressed only through ``BEG_DSN`` (see ``.env.example``), via
:func:`beg.db.get_engine`.
"""

from beg.settings import data_dir, dsn, output_dir, repo_root

__version__ = "0.1.0"

__all__ = ["__version__", "repo_root", "data_dir", "output_dir", "dsn"]
