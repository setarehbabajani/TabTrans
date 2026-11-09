import json

INPUT = "/path/to/Wiki_Input_Selection/tables_tagged.jsonl"
OUTPUT = "/path/to/Wiki_Input_Selection/candidate_tables.jsonl"

MAX_HEADER_ROWS = 2
MIN_ROWS = 5
MAX_COLUMNS = 14
MIN_DISTINCT_TYPES = 4

total_tables = 0
selected_tables = 0

with open(INPUT, encoding="utf-8") as f, open(OUTPUT, "w", encoding="utf-8") as out:
    for line in f:
        total_tables += 1
        table = json.loads(line)

        if table.get("numHeaderRows", 1) > MAX_HEADER_ROWS:
            continue
        if table.get("numDataRows", 0) < MIN_ROWS:
            continue
        if table.get("numCols", 0) > MAX_COLUMNS:
            continue
        if len(set(table["column_tags"])) < MIN_DISTINCT_TYPES:
            continue

        out.write(line)
        selected_tables += 1

print("Total tables:", total_tables)
print("Candidates:", selected_tables)
