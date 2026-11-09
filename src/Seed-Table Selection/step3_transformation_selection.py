import json
import random
from collections import defaultdict

INPUT = "/path/to/Wiki_Input_Selection/candidate_tables.jsonl"
OUTPUT = "/path/to/Wiki_Input_Selection/final_tables.jsonl"

groups = defaultdict(list)
total_tables = 0

with open(INPUT, encoding="utf-8") as f:
    for line in f:
        total_tables += 1
        table = json.loads(line)
        signature = (table.get("numCols", 0), len(set(table["column_tags"])))
        groups[signature].append(line)

selected = [random.choice(groups[signature]) for signature in sorted(groups)]

with open(OUTPUT, "w", encoding="utf-8") as out:
    out.writelines(selected)
