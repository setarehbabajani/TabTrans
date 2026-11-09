import gzip
import json
import re

INPUT = "/path/to/tables.json.gz"
OUTPUT = "/path/to/Wiki_Input_Selection/tables_tagged.jsonl"

PATTERNS = {
    "DATE": re.compile(
        r"\bdate\b|released|publication|published|airdate|premiere|debut|year|age|years|founded|established|born|died|season|term",
        re.I,
    ),
    "PERSON": re.compile(
        r"\bauthor|player|member|ceo|director|president|manager|coach|founder|artist|actor|actress|singer|leader|captain|composer|person|people\b",
        re.I,
    ),
    "LOCATION": re.compile(
        r"\bcountry|city|state|province|region|continent|location|place|venue|stadium|address|capital\b",
        re.I,
    ),
    "CURRENCY": re.compile(
        r"\bprice|cost|revenue|income|budget|salary|earnings|gross\b",
        re.I,
    ),
    "NUMERIC": re.compile(
        r"\bcount|total|number|score|rank|rating|population|points|wins|losses|attendance|votes\b",
        re.I,
    ),
    "UNIT": re.compile(
        r"\bheight|length|distance|weight|km|meter|foot|inch|mile|kg|lb|lbs|ton|celsius|fahrenheit|area|size|unit\b",
        re.I,
    ),
    "ORG": re.compile(
        r"\bcompany|bank|university|college|club|team|airport|party|organization|school|agency\b",
        re.I,
    ),
    "MEDIA": re.compile(
        r"\balbum|song|film|movie|series|episode|journal|magazine|book|novel|track\b",
        re.I,
    ),
    "CODE": re.compile(
        r"\bisbn|issn|iata|icao|swift|sku|symbol|code|unicode|ascii|hex|identifier|id\b",
        re.I,
    ),
    "BOOLEAN": re.compile(
        r"\bactive|enabled|disabled|yes/no|true/false|status\b",
        re.I,
    ),
}

COMPOSITE_ENTITY_PATTERN = re.compile(
    r"\(|\)|\[|\]|\{|\}|,|;|:| - | – | — |/|\\|\sv\s|\|"
)


def tag_header(header):
    for tag, pattern in PATTERNS.items():
        if pattern.search(header):
            return tag
    return "STRING"


def extract_headers(table):
    return [
        cell.get("text", "").strip()
        for row in table.get("tableHeaders", [])
        for cell in row
    ]


def extract_first_rows(table, n_rows=5):
    return [
        [cell.get("text", "").strip() for cell in row]
        for row in table.get("tableData", [])[:n_rows]
    ]


def composite_entity_columns(table):
    rows = extract_first_rows(table)
    if not rows:
        return set()

    num_cols = max(len(r) for r in rows)
    columns = set()
    for col_idx in range(num_cols):
        hits = sum(
            1 for row in rows
            if col_idx < len(row) and COMPOSITE_ENTITY_PATTERN.search(row[col_idx])
        )
        if hits >= 2:
            columns.add(col_idx)
    return columns


with gzip.open(INPUT, "rt", encoding="utf-8") as f, open(OUTPUT, "w", encoding="utf-8") as out:
    for line in f:
        table = json.loads(line)
        headers = extract_headers(table)
        column_tags = [tag_header(h) for h in headers]

        for idx in composite_entity_columns(table):
            if idx < len(column_tags):
                column_tags[idx] = "COMPOSITE_ENTITY"

        table["headers_flat"] = headers
        table["column_tags"] = column_tags
        out.write(json.dumps(table) + "\n")
