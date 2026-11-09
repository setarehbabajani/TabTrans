from .base import Transformation


class PatternCount(Transformation):
    name = "pattern_count"
    input_arity = 1
    output_arity = "n"

    def check(self, columns, tags, params):
        return tags[0] in {"COMPOSITE_ENTITY", "STRING"}

    def apply(self, columns, tags, params):
        col = columns[0]
        tag = tags[0]

        if tag == "COMPOSITE_ENTITY":
            return self._apply_composite_entity(col, params)

        if tag == "STRING":
            return self._apply_string_counts(col)

        return None, None, None

    def _apply_composite_entity(self, col, params):
        delimiter = params.get("delimiter")
        left = params.get("left_wrapper")
        right = params.get("right_wrapper")

        def count_phrases(text):
            text = str(text)
            separators = 0

            if delimiter:
                separators = text.count(delimiter)

            elif left and right:
                separators = min(
                    text.count(left),
                    text.count(right)
                )

            else:
                separators = text.count(";")

            return separators + 1

        new_col = col.apply(count_phrases)

        self.last_used_params = {
            "mode": "composite_entity_phrase_count",
            "delimiter": delimiter,
            "left_wrapper": left,
            "right_wrapper": right,
        }

        return (
            [new_col],
            ["IDENTIFIER_CODE"],
            [{"code_type": "pattern_count", "has_leading_zero": False}],
        )

    def _apply_string_counts(self, col):
        def count_letters(text):
            return sum(1 for ch in str(text) if ch.isalpha())

        def count_words(text):
            return len(str(text).split())

        letters = col.apply(count_letters)
        words = col.apply(count_words)

        self.output_name_hints = ["letters", "words"]

        self.last_used_params = {"mode": "string_letter_word_count"}

        return (
            [letters, words],
            ["IDENTIFIER_CODE", "IDENTIFIER_CODE"],
            [
                {"code_type": "letter_count", "has_leading_zero": False},
                {"code_type": "word_count", "has_leading_zero": False},
            ],
        )
