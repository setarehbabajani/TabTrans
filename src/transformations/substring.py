import pandas as pd
from .base import Transformation


class Substring(Transformation):
    name = "substring"
    input_arity = "n"
    output_arity = "n"

    CITY_ABBREVIATION_DEFAULT_WORD_START = 0
    CITY_ABBREVIATION_DEFAULT_WORD_END = 1

    def check(self, columns, tags, params):
        if (
            len(tags) == 2
            and set(tags) == {"PERSON_NAME_FIRST", "PERSON_NAME_LAST"}
        ):
            return True

        if tags == ["CITY"]:
            return True

        return False

    def apply(self, columns, tags, params):
        if (
            len(tags) == 2
            and set(tags) == {"PERSON_NAME_FIRST", "PERSON_NAME_LAST"}
        ):
            return self._apply_name_initials(columns, tags, params)

        if tags == ["CITY"]:
            return self._apply_city_abbreviation(columns, tags, params)

        return None, None, None

    def _apply_name_initials(self, columns, tags, params):
        first_idx = tags.index("PERSON_NAME_FIRST")
        last_idx = tags.index("PERSON_NAME_LAST")

        first = columns[first_idx].fillna("").astype(str).str.strip()
        last = columns[last_idx].fillna("").astype(str).str.strip()

        def initial(name):
            return name[0].upper() if name else ""

        first_initial = first.apply(initial)
        last_initial = last.apply(initial)

        out_1 = first_initial + ". " + last
        out_2 = first + " " + last_initial + "."
        out_3 = first_initial + "." + last_initial + "."
        out_4 = last + "- " + first_initial + "."

        self.output_name_hints = [
            "fi_last", "first_li", "fi_li", "last_fi"
        ]

        return (
            [out_1, out_2, out_3, out_4],
            [
                "PERSON_NAME_INITIALS",
                "PERSON_NAME_INITIALS",
                "PERSON_NAME_INITIALS",
                "PERSON_NAME_INITIALS",
            ],
            [
                {"format_rule": "first_initial.last"},
                {"format_rule": "first.last_initial"},
                {"format_rule": "first_initial.last_initial"},
                {"format_rule": "last-first_initial"},
            ],
        )

    def _apply_city_abbreviation(self, columns, tags, params):
        col = columns[0]

        col_params = (
            params[0]
            if isinstance(params, list) and params
            else (params if isinstance(params, dict) else {})
        )

        word_start = col_params.get(
            "word_start", self.CITY_ABBREVIATION_DEFAULT_WORD_START
        )
        word_end = col_params.get(
            "word_end", self.CITY_ABBREVIATION_DEFAULT_WORD_END
        )

        self.last_used_params = {
            "word_start": word_start,
            "word_end": word_end,
        }

        def abbreviate(city):
            if pd.isna(city):
                return None

            words = str(city).strip().split()
            if not words:
                return None

            start = max(0, word_start)
            end = min(len(words) - 1, word_end)

            if start > end:
                return None

            selected = words[start:end + 1]
            initials = " ".join(w[0].upper() for w in selected if w)

            return initials or None

        new_col = col.apply(abbreviate)

        return (
            [new_col],
            ["CITY_ABBREVIATION"],
            [{}],
        )
