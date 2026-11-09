import re
from .base import Transformation


class PersonDerivations(Transformation):
    name = "person_derivations"
    input_arity = 1
    output_arity = "n"

    EMAIL_DOMAIN = "ualberta.ca"
    URL_BASE = "http://www.ualberta.ca/~"

    ALLOWED_IDENTITY_TAGS = {"PERSON_USERNAME", "PERSON_NAME_INITIALS"}

    TITLE = "Dr."
    TITLE_DELIMITER = " "

    def check(self, columns, tags, params):
        if len(tags) != 1:
            return False

        return (
            tags[0] in self.ALLOWED_IDENTITY_TAGS
            or tags[0] == "PERSON_NAME_FULL"
        )

    def apply(self, columns, tags, params):
        tag = tags[0]

        if tag in self.ALLOWED_IDENTITY_TAGS:
            return self._apply_identity(columns, tags, params)

        if tag == "PERSON_NAME_FULL":
            return self._apply_title(columns, tags, params)

        return None, None, None

    def _sanitize(self, value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    def _apply_identity(self, columns, tags, params):
        identifier = columns[0].astype(str).apply(self._sanitize)

        email = identifier + "@" + self.EMAIL_DOMAIN
        url = self.URL_BASE + identifier

        self.output_name_hints = ["email", "url"]

        return (
            [email, url],
            ["EMAIL_FULL", "URL"],
            [{}, {}],
        )

    def _apply_title(self, columns, tags, params):
        col = columns[0]

        new_col = col.apply(
            lambda name: (
                f"{self.TITLE}{self.TITLE_DELIMITER}{name}"
                if name is not None and str(name).strip() != ""
                else name
            )
        )

        return (
            [new_col],
            ["PERSON_NAME_WITH_TITLE"],
            [{"title": self.TITLE, "delimiter": self.TITLE_DELIMITER}],
        )
