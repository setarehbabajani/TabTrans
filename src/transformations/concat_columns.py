import pandas as pd
from .base import Transformation
import re


class ConcatColumns(Transformation):
    name = "concat_columns"
    input_arity = "n"
    output_arity = 1

    def check(self, columns, tags, params):
        tag_set = set(tags)

        if tag_set.issubset({
            "PERSON_NAME_FIRST",
            "PERSON_NAME_MIDDLE",
            "PERSON_NAME_LAST"
        }) and len(tags) >= 2:
            split_roots = []
            for col, tag in zip(columns, tags):
                if tag not in {
                    "PERSON_NAME_FIRST",
                    "PERSON_NAME_MIDDLE",
                    "PERSON_NAME_LAST"
                }:
                    continue
                name = col.name
                if re.match(r"^.+_\d+$", name):
                    split_roots.append(name.rsplit("_", 1)[0])
            if split_roots:
                if len(set(split_roots)) > 1:
                    print("cannot merge different split roots!")
                    return False
            return True

        if tag_set == {"YEAR"} and len(tags) == 2:
            return True

        if tag_set == {"YEAR", "MONTH", "DAY"}:
            split_roots = []
            for col, tag in zip(columns, tags):
                if tag not in {
                    "YEAR",
                    "MONTH",
                    "DAY"
                }:
                    continue
                name = col.name
                if re.match(r"^.+_\d+$", name):
                    split_roots.append(name.rsplit("_", 1)[0])
            if split_roots:
                if len(set(split_roots)) > 1:
                    print("cannot merge different split roots!")
                    return False
            return True
        return False

    def apply(self, columns, tags, params):
        df = pd.concat(columns, axis=1)
        tag_set = set(tags)

        if tag_set.issubset({
            "PERSON_NAME_FIRST",
            "PERSON_NAME_MIDDLE",
            "PERSON_NAME_LAST"
        }):
            new_col = df.fillna("").astype(str).agg(
                lambda x: " ".join([v for v in x if v.strip() != ""]),
                axis=1
            )
            return [new_col], ["PERSON_NAME_FULL"], [{"delimiter": " "}]

        if tag_set == {"YEAR"} and len(tags) == 2:
            new_col = df.fillna("").astype(str).agg("-".join, axis=1)
            new_params = [{"delimiter": "-"}]
            return [new_col], ["YEAR_RANGE"], new_params

        if tag_set == {"YEAR", "MONTH", "DAY"}:
            df = df.astype(str)

            new_col = (
                df.iloc[:,0].str.zfill(4) + "-" +
                df.iloc[:,1].str.zfill(2) + "-" +
                df.iloc[:,2].str.zfill(2)
            )

            return [new_col], ["DATE_ISO_YYYY_MM_DD"], [{"delimiter": "-"}]
