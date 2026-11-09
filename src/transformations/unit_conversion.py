import re
import unicodedata
from .base import Transformation


class UnitConversion(Transformation):
    name = "unit_conversion"
    input_arity = 1
    output_arity = 1

    EXCHANGE_RATES = {
        ("USD", "EUR"): 0.86,
        ("EUR", "USD"): 1.17,
        ("USD", "CAD"): 1.38,
        ("CAD", "USD"): 0.72,
        ("CAD", "INR"): 0.014,
    }

    SYMBOL_TO_CODE = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "C$": "CAD",
        "₹": "INR"
    }

    CODE_TO_SYMBOL = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "CAD": "C$",
        "INR": "₹"
    }

    LENGTH_BASE = "m"
    MASS_BASE = "g"
    TEMP_BASE = "C"
    DURATION_BASE = "second"
    BASE_CODE = "CAD"
    BASE_SYMBOL = "C$"

    LENGTH_FACTORS = {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1,
        "km": 1000,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.34,
        "ha": 100,
    }

    MASS_FACTORS = {
        "mg": 0.001,
        "g": 1,
        "kg": 1000,
        "ton": 1_000_000,
        "oz": 28.3495,
        "lb": 453.592
    }

    DURATION_FACTORS = {
        "second": 1,
        "seconds": 1,
        "sec": 1,
        "minute": 60,
        "minutes": 60,
        "min": 60,
        "hour": 3600,
        "hours": 3600,
        "day": 86400,
        "days": 86400,
        "week": 604800,
        "weeks": 604800,
        "year": 31557600,
        "years": 31557600
    }

    def check(self, columns, tags, params):
        tag = tags[0]

        if tag == "STRING":
            return params.get("has_unicode") is True

        if tag == "NUM_WITH_UNIT":
            unit_type = params.get("unit_type")
            unit = params.get("unit")
            if not unit_type or not unit:
                return False

            if unit_type in {"length_metric", "length_imperial"}:
                return (
                    unit in self.LENGTH_FACTORS
                    and unit != self.LENGTH_BASE
                )
            if unit_type == "mass":
                return (
                    unit in self.MASS_FACTORS
                    and unit != self.MASS_BASE
                )
            if unit_type == "temperature":
                return unit in {"F", "K"}
            if unit_type == "duration":
                return (
                    unit in self.DURATION_FACTORS
                    and unit != self.DURATION_BASE
                )

        if tag == "CURRENCY_VALUE":
            symbol = params.get("currency_symbol")
            code = params.get("currency_code")

            if not code and symbol:
                code = self.SYMBOL_TO_CODE.get(symbol)

            if not code:
                return False

            return code != self.BASE_CODE

        return False

    def parse(self, text):
        text = str(text).strip()
        match = re.match(
            r"^([+-]?[0-9,]*\.?[0-9]+)(?:\s*(.*))?$",
            text
        )

        if not match:
            return None, None

        try:
            value = float(
                match.group(1).replace(",", "")
            )
        except:
            return None, None

        unit = match.group(2)
        if unit is not None:
            unit = unit.strip()

            if unit == "":
                unit = None

        return value, unit

    def parse_currency(self, text):
        text = str(text).strip()

        match = re.match(r"([^\d\-\.]+)?\s*([0-9,]*\.?[0-9]+)", text)
        if not match:
            return None, None, None

        prefix = match.group(1)
        value = match.group(2)

        try:
            value = float(value.replace(",", ""))
        except:
            return None, None, None

        code = None
        symbol = None

        if prefix:
            prefix = prefix.strip()

            if prefix in self.SYMBOL_TO_CODE:
                symbol = prefix
                code = self.SYMBOL_TO_CODE[prefix]
            elif prefix in self.CODE_TO_SYMBOL:
                code = prefix
                symbol = self.CODE_TO_SYMBOL[prefix]

        return symbol, code, value

    def unicode_to_ascii(self, text):
        return unicodedata.normalize(
            "NFKD",
            str(text)
        ).encode(
            "ascii",
            "ignore"
        ).decode("ascii")

    def apply(self, columns, tags, params):
        if params is None:
            params = [{} for _ in tags]
        params = params.copy() if params else {}

        col = columns[0]
        tag = tags[0]
        if tag == "STRING" and params.get("has_unicode") is True:
            col = col.apply(self.unicode_to_ascii)

            out_params = dict(params)
            out_params["has_unicode"] = False

            return [col], ["STRING"], [out_params]

        if tag == "CURRENCY_VALUE":
            source_symbol = params.get("currency_symbol")
            source_code = params.get("currency_code")
            if not source_code and source_symbol:
                source_code = self.SYMBOL_TO_CODE.get(source_symbol)

            if source_code == self.BASE_CODE:
                return [col], tags, params

            target_code = self.BASE_CODE
            target_symbol = self.BASE_SYMBOL

            rate = self.EXCHANGE_RATES.get((source_code, target_code))

            if rate is None:
                return [col], tags, params

            def convert(x):
                symbol, code, value = self.parse_currency(x)
                if value is None:
                    return x

                row_code = code or source_code

                if row_code == target_code:
                    return x

                rate_local = self.EXCHANGE_RATES.get((row_code, target_code))
                if rate_local is None:
                    return x

                new_val = value * rate_local
                return f"{target_symbol} {new_val:.4f}"

            col = col.apply(convert)

            params["currency_symbol"] = target_symbol
            params["currency_code"] = target_code

            return [col], ["CURRENCY_VALUE"], [params]

        if tag == "NUM_WITH_UNIT":
            unit_type = params["unit_type"]
            unit = params["unit"]
            squared = params.get("is_squared", False)

            if unit_type in {"length_metric", "length_imperial"}:
                factor = self.LENGTH_FACTORS[unit]
                target_unit = self.LENGTH_BASE

                if squared:
                    factor = factor ** 2
                    target_unit = f"{target_unit}²"

                def convert(x):
                    value, u = self.parse(x)
                    if value is None:
                        return x
                    new_val = value * factor
                    return f"{new_val:.4f} {target_unit}"

                col = col.apply(convert)
                params["unit"] = target_unit
                return [col], ["NUM_WITH_UNIT"], [params]

            if unit_type == "mass":
                factor = self.MASS_FACTORS[unit]
                target_unit = self.MASS_BASE
                if squared:
                    factor = factor ** 2
                    target_unit = f"{target_unit}²"

                def convert(x):
                    value, u = self.parse(x)
                    if value is None:
                        return x
                    new_val = value * factor
                    return f"{new_val:.4f} {target_unit}"

                col = col.apply(convert)
                params["unit"] = target_unit
                return [col], ["NUM_WITH_UNIT"], [params]

            if unit_type == "temperature":
                def convert(x):
                    value, u = self.parse(x)
                    if value is None:
                        return x
                    if unit == "F":
                        new_val = (value - 32) * 5 / 9
                    elif unit == "K":
                        new_val = value - 273.15
                    else:
                        new_val = value
                    return f"{new_val:.4f} {self.TEMP_BASE}"

                col = col.apply(convert)
                params["unit"] = self.TEMP_BASE
                return [col], ["NUM_WITH_UNIT"], [params]

            if unit_type == "duration":
                factor = self.DURATION_FACTORS[unit]
                if factor is None:
                    print(f"Unknown duration unit: {unit}")
                    return [col], tags, params

                def convert(x):
                    value, u = self.parse(x)
                    if value is None:
                        return x
                    new_val = value * factor
                    return f"{new_val:.4f} {self.DURATION_BASE}"

                col = col.apply(convert)
                params["unit"] = self.DURATION_BASE
                return [col], ["NUM_WITH_UNIT"], [params]

        return [col], tags, params
