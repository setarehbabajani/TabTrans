import csv
import json
from pathlib import Path

INPUT = "/path/to/Wiki_Input_Selection/final_tables.jsonl"
OUTPUT_DIR = "/path/to/wiki_tables"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

with open(INPUT, encoding="utf-8") as f:
    for line in f:
        table = json.loads(line)
        table_dir = Path(OUTPUT_DIR) / f"wiki_{table['_id']}"
        table_dir.mkdir(exist_ok=True)

        rows = [[cell["text"] for cell in row] for row in table.get("tableHeaders", [])]
        rows += [[cell["text"] for cell in row] for row in table.get("tableData", [])]

        with open(table_dir / "source.csv", "w", newline="", encoding="utf-8") as out:
            csv.writer(out).writerows(rows)

print("Done")
