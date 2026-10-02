"""Loading epJSON models and weather files into the ``ep`` schema.

The stage runs in this order, and each module owns one step of it:

ep_schema              parse ``Energy+.schema.epJSON``; pre-scan the model set
tables                 define ``ep_model`` and one table set per object type
loader                 read epJSON files into those tables
foreign_keys           foreign keys for unambiguous label references
registries             registry tables for ambiguous ones
postprocess            prune, coerce booleans, one-hot encode
weather                EPW and DDY files -> ``ep.weather_city``

Supporting modules:

identifiers            SQL-safe table, column and constraint names
schema_map             EnergyPlus schema field -> SQLAlchemy type and constraints
models                 model registry rows, upserts, epJSON loading
foreign_keys_advanced  candidate scoring for ambiguous references

The heavier entry points are not re-exported here, so that ``beg.ingest`` stays
cheap to import; import them from their own modules.
"""

from beg.ingest.foreign_keys import (
    add_label_based_fks,
)
from beg.ingest.foreign_keys_advanced import (
    candidate_targets_for_labels,
    candidate_targets_set,
    fully_qualified,
    list_existing_fk_names,
    should_queue_index,
    table_row_estimates,
)
from beg.ingest.identifiers import (
    ID_MAX,
    PG_KEYWORDS,
    PG_SYSTEM_COLS,
    qident,
    safe_column_name,
    safe_constraint_name,
    safe_table_name,
    sanitize,
    shorten_identifier,
)
from beg.ingest.models import (
    build_scalar_field_map,
    count_instances_in_container,
    ep_model_table,
    insert_main_return_id,
    insert_model,
    insert_or_upsert_main,
    load_epjson,
    normalize_instances_container,
    portable_file_path,
    replace_children,
    row_data_from_fields,
    sha256_file,
    upsert_main_return_id,
)
from beg.ingest.schema_map import (
    build_check_enum,
    build_check_minmax,
    classify_fields,
    get,
    return_as_list,
    sa_type_from_schema,
    server_default_to_str,
    set_server_default,
)

__all__ = [
    # identifiers
    "ID_MAX",
    "PG_KEYWORDS",
    "PG_SYSTEM_COLS",
    "sanitize",
    "shorten_identifier",
    "safe_table_name",
    "safe_column_name",
    "safe_constraint_name",
    "qident",
    # schema_map
    "build_check_enum",
    "build_check_minmax",
    "classify_fields",
    "get",
    "return_as_list",
    "sa_type_from_schema",
    "server_default_to_str",
    "set_server_default",
    # models
    "ep_model_table",
    "sha256_file",
    "portable_file_path",
    "insert_model",
    "upsert_main_return_id",
    "insert_main_return_id",
    "load_epjson",
    "normalize_instances_container",
    "build_scalar_field_map",
    "row_data_from_fields",
    "insert_or_upsert_main",
    "replace_children",
    "count_instances_in_container",
    # foreign keys
    "add_label_based_fks",
    "list_existing_fk_names",
    "table_row_estimates",
    "candidate_targets_set",
    "candidate_targets_for_labels",
    "fully_qualified",
    "should_queue_index",
]
