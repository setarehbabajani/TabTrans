from src.transformations.date_format import DateFormat
from src.transformations.unit_conversion import UnitConversion
from src.transformations.split_column import SplitColumn
from src.transformations.concat_columns import ConcatColumns
from src.transformations.numeric_formatting import NumericFormatting
from src.transformations.numeric_transformation import NumericTransformation
from src.transformations.boolean_normalization import BooleanNormalization
from src.transformations.pattern_count import PatternCount
from src.transformations.string_normalization import StringNormalization
from src.transformations.substring import Substring
from src.transformations.person_derivations import PersonDerivations
from src.transformations.structural import (
    sort_rows,
    drop_columns,
    drop_duplicates,
    wrap_multi_header,
    transpose,
    unpivot,
    groupby_aggregate,
    pivot,
    drop_rows,
    divide_mixed,
)

STRUCTURAL_MODULES = (
    sort_rows,
    drop_columns,
    drop_duplicates,
    wrap_multi_header,
    transpose,
    unpivot,
    groupby_aggregate,
    pivot,
    drop_rows,
    divide_mixed,
)


def load_transformations():
    return [
        DateFormat(),
        UnitConversion(),
        SplitColumn(),
        ConcatColumns(),
        NumericFormatting(),
        NumericTransformation(),
        BooleanNormalization(),
        PatternCount(),
        StringNormalization(),
        Substring(),
        PersonDerivations(),
    ]
