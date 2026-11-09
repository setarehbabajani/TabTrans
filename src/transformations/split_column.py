import codecs
import pandas as pd
import re
from .base import Transformation

_DASH_VARIANTS = (
    "\u002d",
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2212",
)


def normalize_delimiter(delimiter):
    if delimiter is None or not isinstance(delimiter, str):
        return delimiter
    if "\\u" in delimiter.lower():
        try:
            return codecs.decode(delimiter, "unicode_escape")
        except (UnicodeError, ValueError):
            pass
    return delimiter


_DASH_SPLIT_RE = r"[-\u2010\u2011\u2012\u2013\u2014\u2212]"


def _delimiter_score(delimiter, sample):
    hits = sample[sample.str.contains(re.escape(delimiter), regex=False)]
    if hits.empty:
        return (0.0, 0)
    ok = sum(1 for v in hits if len(v.split(delimiter)) >= 2)
    return (ok / len(hits), ok)


def resolve_delimiter(delimiter, series, *, default=None):
    delimiter = normalize_delimiter(
        delimiter if delimiter is not None else default
    )
    if delimiter is None:
        return delimiter

    sample = series.dropna().astype(str)
    if sample.empty:
        return delimiter

    if delimiter not in _DASH_VARIANTS:
        return delimiter

    present = [
        variant
        for variant in _DASH_VARIANTS
        if sample.str.contains(re.escape(variant), regex=False).any()
    ]
    if not present:
        return delimiter

    best = delimiter
    best_score = (-1.0, -1)
    for variant in present:
        score = _delimiter_score(variant, sample)
        if score > best_score:
            best_score = score
            best = variant

    return best


def _extract_wrapped_parts(value, left, right):
    if left not in value or right not in value:
        return None

    start = value.find(left)
    end = value.find(right, start + len(left))
    if end == -1:
        return None

    prefix = value[:start].strip()
    inside = value[start + len(left): end].strip()
    suffix = value[end + len(right):].strip()
    outside_parts = [part for part in (prefix, suffix) if part]
    outside = " ".join(outside_parts)
    return outside, inside


def _string_series(series):
    na_mask = series.isna()
    out = series.astype(str)
    return out.mask(na_mask, "")

SPLIT_DELIMITERS = {
    "PERSON_NAME_FULL": " ",
    "DATE_ISO_YYYY_MM_DD": "-",
    "EMAIL_FULL": "@",
    "YEAR_RANGE": "-",
    "ADDRESS_FULL": ",",
}

class SplitColumn(Transformation):
    name = "split_column"
    input_arity = 1
    output_arity = "n"

    def check(self, columns, tags, params):
        tag = tags[0]

        allowed = {
            "PERSON_NAME_FULL",
            "COMPOSITE_ENTITY",
            "YEAR_RANGE",
            "ADDRESS_FULL",
            "DATE_ISO_YYYY_MM_DD",
            "EMAIL_FULL"
        }

        return tag in allowed

    def apply(self, columns, tags, params):
        col = columns[0]
        tag = tags[0]
        self.last_used_params = {
            "delimiter": params.get("delimiter") or SPLIT_DELIMITERS.get(tag),
        }
        if params.get("left_wrapper") is not None:
            self.last_used_params["left_wrapper"] = params["left_wrapper"]
        if params.get("right_wrapper") is not None:
            self.last_used_params["right_wrapper"] = params["right_wrapper"]

        if tag == "PERSON_NAME_FULL":
            split = _string_series(col).str.split()

            def is_initial_token(w):
                parts = w.split(".")
                parts = [p for p in parts if p]

                return all(len(p) == 1 for p in parts)

            cleaned = split.apply(
                lambda x: [
                    w for w in x
                    if not (
                        w.endswith(".")
                        and not is_initial_token(w)
                    )
                ]
            )

            cleaned = cleaned.apply(
                lambda x: [w.rstrip(".") for w in x]
            )

            first = cleaned.str[0]
            last = cleaned.str[-1]

            has_middle = cleaned.apply(
                lambda x: len(x) > 2
            )

            if has_middle.any():
                middle = cleaned.apply(
                    lambda x:
                    " ".join(x[1:-1])
                    if len(x) > 2
                    else None
                )

                return (
                    [first, middle, last],
                    [
                        "PERSON_NAME_FIRST",
                        "PERSON_NAME_MIDDLE",
                        "PERSON_NAME_LAST"
                    ],
                    [{"delimiter": " "}, {"delimiter": " "}, {"delimiter": " "}]
                )

            else:
                return (
                    [first, last],
                    [
                        "PERSON_NAME_FIRST",
                        "PERSON_NAME_LAST"
                    ],
                    [{"delimiter": " "}, {"delimiter": " "}]
                )

        if tag == "COMPOSITE_ENTITY":
            delimiter = resolve_delimiter(
                params.get("delimiter"), col
            )
            left = params.get("left_wrapper")
            right = params.get("right_wrapper")

            series = _string_series(col)

            cols = []

            if delimiter and left and right:
                split = series.str.split(
                    delimiter,
                    regex=False,
                    expand=True
                )

                for i in range(split.shape[1]):
                    part = _string_series(split[i])

                    outside_vals = []
                    inside_vals = []

                    for value in part:
                        extracted = _extract_wrapped_parts(
                            value, left, right
                        )
                        if extracted is not None:
                            outside_vals.append(extracted[0])
                            inside_vals.append(extracted[1])
                            continue

                        outside_vals.append(
                            value.strip()
                        )

                        inside_vals.append(None)

                    outside = pd.Series(outside_vals)
                    inside = pd.Series(inside_vals)

                    outside_clean = outside.where(
                        outside != "",
                        None
                    )

                    inside_clean = inside.where(
                        inside != "",
                        None
                    )

                    if outside_clean.notna().any():
                        cols.append(outside_clean)

                    if inside_clean.notna().any():
                        cols.append(inside_clean)

            elif delimiter:
                split = series.str.split(
                    delimiter,
                    regex=False,
                    expand=True
                )

                temp_cols = [
                    _string_series(split[i]).str.strip()
                    for i in range(split.shape[1])
                ]

                for c in temp_cols:
                    if (
                        delimiter != ","
                        and c.str.contains(",", regex=False).any()
                    ):
                        sub_split = c.str.split(
                            ",",
                            regex=False,
                            expand=True
                        )

                        for j in range(sub_split.shape[1]):
                            cols.append(
                                _string_series(sub_split[j])
                                .str.strip()
                            )

                    else:
                        cols.append(c)

            elif left and right:
                outside_vals = []
                inside_vals = []

                for value in series:
                    extracted = _extract_wrapped_parts(
                        value, left, right
                    )
                    if extracted is not None:
                        outside_vals.append(extracted[0])
                        inside_vals.append(extracted[1])
                        continue

                    outside_vals.append(
                        value.strip()
                    )

                    inside_vals.append(None)

                cols = [
                    pd.Series(outside_vals),
                    pd.Series(inside_vals)
                ]

            else:
                split = series.str.split(
                    r"[;:\-()\[\]]",
                    expand=True
                )

                for i in range(split.shape[1]):
                    cols.append(
                        _string_series(split[i]).str.strip()
                    )

            new_tags = [None] * len(cols)
            new_params = []
            for _ in cols:
                new_params.append({
                    "delimiter": delimiter,
                    "left_wrapper": left,
                    "right_wrapper": right
                })

            return cols, new_tags, new_params

        if tag == "YEAR_RANGE":
            cleaned = col.astype(str).str.replace(
                r"[()\s]",
                "",
                regex=True
            )

            delimiter = resolve_delimiter(
                params.get("delimiter"),
                cleaned,
                default="-",
            )

            if delimiter in _DASH_VARIANTS:
                split = cleaned.str.split(
                    _DASH_SPLIT_RE,
                    regex=True,
                    expand=True,
                )
            else:
                split = cleaned.str.split(
                    delimiter,
                    regex=False,
                    expand=True,
                )

            if split.shape[1] < 2:
                return None, None, None

            return (
                [
                    split[0].str.strip(),
                    split[1].str.strip()
                ],
                ["YEAR", "YEAR"],
                [{"delimiter": delimiter}, {"delimiter": delimiter}]
            )

        if tag == "ADDRESS_FULL":
            delimiter = resolve_delimiter(
                params.get("delimiter"),
                col,
                default=",",
            )

            split = col.astype(str).str.split(
                delimiter,
                regex=False,
                expand=True
            )

            outputs = []
            tags_out = []
            params_out = []

            idx = 0

            if (
                params.get("has_street")
                and split.shape[1] > idx
            ):
                outputs.append(
                    split[idx].str.strip()
                )

                tags_out.append("STREET")
                params_out.append({"delimiter": delimiter})

                idx += 1

            if (
                params.get("has_city")
                and split.shape[1] > idx
            ):
                outputs.append(
                    split[idx].str.strip()
                )

                tags_out.append("CITY")
                params_out.append({"delimiter": delimiter})

                idx += 1

            if (
                params.get("has_state")
                and split.shape[1] > idx
            ):
                outputs.append(
                    split[idx].str.strip()
                )

                tags_out.append("STATE")
                params_out.append({"delimiter": delimiter})

                idx += 1

            if (
                params.get("has_country")
                and split.shape[1] > idx
            ):
                outputs.append(
                    split[idx].str.strip()
                )

                tags_out.append("COUNTRY_NAME")
                params_out.append({"delimiter": delimiter})

            return outputs, tags_out, params_out

        if tag == "DATE_ISO_YYYY_MM_DD":
            split = col.astype(str).str.split(
                "-",
                regex=False,
                expand=True
            )

            if split.shape[1] < 3:
                return None, None, None

            return (
                [
                    split[0].str.strip(),
                    split[1].str.strip(),
                    split[2].str.strip()
                ],
                ["YEAR", "MONTH", "DAY"],
                [{"delimiter": "-"}, {"delimiter": "-"}, {"delimiter": "-"}]
            )

        if tag == "EMAIL_FULL":
            domain = (
                col.astype(str)
                .str.split("@", regex=False)
                .str[-1]
            )

            return (
                [domain],
                ["EMAIL_DOMAIN"],
                [{"delimiter": "@"}]
            )
