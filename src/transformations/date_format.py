import pandas as pd
import jdatetime
import re
from datetime import datetime
from .base import Transformation


class DateFormat(Transformation):
    name = "date_format"
    input_arity = 1
    output_arity = 1

    def check(self, columns, tags, params):
        tag = tags[0]

        allowed_inputs = {
            "DATE_LONG_MONTH_NAME",
            "DATE_DD_MM_YYYY",
            "DATE_MM_DD_YYYY"
        }

        if tag in allowed_inputs:
            return True

        if tag == "YEAR":
            return params.get("calendar_system") != "jalali"

        if tag == "DATE_ISO_YYYY_MM_DD":
            return params.get("calendar_system") != "jalali"

        return False

    def apply(self, columns, tags, params):
        col = columns[0]
        tag = tags[0]

        if tag == "DATE_LONG_MONTH_NAME":
            is_month_first = params.get(
                "is_month_first",
                True
            )

            current_year = datetime.now().year

            def normalize_date(x):
                if pd.isna(x):
                    return None

                x = str(x).strip()

                if x == "":
                    return None

                if not re.search(r"\d{4}", x):
                    x = f"{x} {current_year}"

                formats = []

                if is_month_first:
                    formats.extend([
                        "%B %d %Y",
                        "%b %d %Y",
                        "%B %d, %Y",
                        "%b %d, %Y",
                        "%Y %B %d",
                        "%Y %b %d"
                    ])

                else:
                    formats.extend([
                        "%d %B %Y",
                        "%d %b %Y",
                        "%d %B, %Y",
                        "%d %b, %Y",
                        "%Y %d %B",
                        "%Y %d %b"
                    ])

                for fmt in formats:
                    try:
                        parsed = datetime.strptime(
                            x,
                            fmt
                        )

                        return parsed.strftime(
                            "%Y-%m-%d"
                        )

                    except Exception:
                        continue

                try:
                    parsed = pd.to_datetime(
                        x,
                        errors="coerce"
                    )

                    if pd.notna(parsed):
                        return parsed.strftime(
                            "%Y-%m-%d"
                        )

                except Exception:
                    pass

                return None

            col = col.apply(normalize_date)

            self.last_used_params = {"is_month_first": is_month_first}

            return (
                [col],
                ["DATE_ISO_YYYY_MM_DD"],
                [{"calendar_system": "gregorian"}]
            )

        if tag == "DATE_DD_MM_YYYY":
            delimiter = params.get(
                "delimiter",
                "-"
            )

            def convert(x):
                try:
                    if pd.isna(x):
                        return None

                    parts = str(x).split(
                        delimiter
                    )

                    if len(parts) < 3:
                        return None

                    day = int(parts[0])
                    month = int(parts[1])
                    year = int(parts[2])

                    return (
                        f"{year:04d}-"
                        f"{month:02d}-"
                        f"{day:02d}"
                    )

                except Exception:
                    return None

            col = col.apply(convert)

            self.last_used_params = {"delimiter": delimiter}

            return (
                [col],
                ["DATE_ISO_YYYY_MM_DD"],
                [{"calendar_system": "gregorian"}]
            )

        if tag == "DATE_MM_DD_YYYY":
            delimiter = params.get(
                "delimiter",
                "-"
            )

            def convert(x):
                try:
                    if pd.isna(x):
                        return None

                    parts = str(x).split(
                        delimiter
                    )

                    if len(parts) < 3:
                        return None

                    month = int(parts[0])
                    day = int(parts[1])
                    year = int(parts[2])

                    return (
                        f"{year:04d}-"
                        f"{month:02d}-"
                        f"{day:02d}"
                    )

                except Exception:
                    return None

            col = col.apply(convert)

            self.last_used_params = {"delimiter": delimiter}

            return (
                [col],
                ["DATE_ISO_YYYY_MM_DD"],
                [{"calendar_system": "gregorian"}]
            )

        if tag == "YEAR":
            def convert_year(y):
                try:
                    if pd.isna(y):
                        return None

                    g = int(y)

                    j = jdatetime.date.fromgregorian(
                        date=pd.Timestamp(
                            year=g,
                            month=1,
                            day=1
                        )
                    )

                    return str(j.year)

                except Exception:
                    return None

            col = col.apply(convert_year)

            self.last_used_params = {
                "calendar_system_in": "gregorian",
                "calendar_system_out": "jalali",
            }

            return (
                [col],
                ["YEAR"],
                [{"calendar_system": "jalali"}]
            )

        if tag == "DATE_ISO_YYYY_MM_DD":
            def convert_date(x):
                try:
                    if pd.isna(x):
                        return None

                    g = pd.to_datetime(x)

                    if pd.isna(g):
                        return None

                    j = jdatetime.date.fromgregorian(
                        date=g.date()
                    )

                    return (
                        f"{j.year:04d}-"
                        f"{j.month:02d}-"
                        f"{j.day:02d}"
                    )

                except Exception:
                    return None

            col = col.apply(convert_date)

            self.last_used_params = {
                "calendar_system_in": "gregorian",
                "calendar_system_out": "jalali",
            }

            return (
                [col],
                ["DATE_ISO_YYYY_MM_DD"],
                [{"calendar_system": "jalali"}]
            )
