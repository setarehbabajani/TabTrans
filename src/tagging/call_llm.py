import os
import re
import json
import urllib.error
import urllib.request
import pandas as pd
from src.pipeline.array_config import project_root

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"
THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "8192"))
MAX_TAG_COLUMNS = int(os.environ.get("TAG_MAX_COLUMNS", "40"))

ROOT = project_root()
INPUT_DIR = "/path to source tables"
CATALOG_FILE = str(ROOT / "src" / "tagging" / "catalog.yaml")
FINAL_TABLES_JSONL = "/path to jsnol file of seed table selection stage"

PROMPT_TEMPLATE = """
You are a data schema analysis assistant.

Your task is to analyze a table and assign a semantic tag to each column based on a provided tag catalog.

The table information provided to you includes:
1. The header row (column names)
2. 10 sample rows
3. A catalog of allowed tags

TASK RULES:
1. Each column MUST receive exactly one tag. Each columned should be tagged independently based on its own header and sample values. Use both header names and cell values.
2. Tags must come only from the catalog. Do not invent new tags.
3. If multiple tags are possible, choose the most specific one.
4. Always choose the MOST SPECIFIC semantic tag possible. Use STRING only as a last resort.
6. for NUM_WITH_UNIT, If the unit cannot be inferred exactly, infer the most likely type using the allowed_units and do not leave it empty!
7. For delimiters, use the EXACT character in the sample values (e.g. en-dash – in "1855–1866", not ASCII hyphen -). Put the character itself in JSON, not a \\u2013 escape string.
8. NUMERIC should ONLY be used when ALL sample values are numeric. If any value contains letters (not null or N/A), do NOT use NUMERIC.
9. Treat the following values as null: "", N/A, NA, NULL, NONE, -, --, ---, _, __, ___, ?, ??. Empty cells or null values do NOT count as a separate data type. So, if a column has, just see the other rows and put the tag. If all were empty, put "STRING".

MULTI HEADER:
Do NOT infer or analyze multi-header structure. Always set "multi_header": False.


OUTPUT FORMAT: Return ONLY valid JSON:

{{
  "multi_header": False,
  "columns": [
    {{
      "column_name": "...",
      "tag": "TAG_NAME",
      "parameters": {{
        "param1": value,
        "param2": value
      }}
    }}
  ]
}}

PARAMETER RULES:
- If the tag has defined parameters in the catalog, ALL of them MUST be filled. If the tag has NO parameters, return an EMPTY object.
- All required parameters must be fully populated. DO NOT omit parameters. DO NOT leave required parameters empty or null!
- Do not add parameters that are not defined in the catalog.

TAG CATALOG: {catalog}

TABLE HEADER: {header}

TABLE SAMPLE ROWS: {rows}
"""


class WikiTaggingFailed(RuntimeError):
    pass


def load_catalog():
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        return f.read()


def call_gemini(prompt):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL_NAME}:generateContent?key={API_KEY}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "thinkingConfig": {"thinkingBudget": THINKING_BUDGET},
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise WikiTaggingFailed(f"Gemini HTTP {e.code}: {detail}") from e
    except Exception as e:
        raise WikiTaggingFailed(f"Gemini call failed: {e}") from e

    candidates = payload.get("candidates") or []
    if not candidates:
        raise WikiTaggingFailed("Gemini returned no candidates")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(
        p["text"] for p in parts if p.get("text") and not p.get("thought")
    ).strip()
    if not text:
        raise WikiTaggingFailed("Gemini returned no answer text")
    return text


def call_llm_tagging(df: pd.DataFrame, wiki_folder=None):
    if len(df.columns) > MAX_TAG_COLUMNS:
        raise WikiTaggingFailed(
            f"{wiki_folder or ''}: {len(df.columns)} columns, limit is {MAX_TAG_COLUMNS}"
        )

    prompt = PROMPT_TEMPLATE.format(
        catalog=load_catalog(),
        header=json.dumps(list(df.columns), indent=2),
        rows=json.dumps(df.iloc[1:11].astype(str).values.tolist(), indent=2),
    )

    text = call_gemini(prompt)
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text).strip()

    try:
        result = json.loads(text)
    except Exception as e:
        raise WikiTaggingFailed(
            f"{wiki_folder or ''}: reply was not JSON ({e}): {text[:1500]!r}"
        ) from e

    columns = result.get("columns") if isinstance(result, dict) else None
    if not isinstance(columns, list):
        raise WikiTaggingFailed(f"{wiki_folder or ''}: reply had no columns list")

    for col in columns:
        if isinstance(col, dict) and "parameters" not in col:
            col["parameters"] = {}

    return result


def load_header_counts():
    header_counts = {}
    with open(FINAL_TABLES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            table = json.loads(line)
            header_counts[str(table["_id"])] = table.get("numHeaderRows", 1)
    return header_counts


def merge_two_headers(df):
    header1 = [str(x).strip() for x in df.iloc[0]]
    header2 = [str(x).strip() for x in df.iloc[1]]

    new_cols = []
    for h, v in zip(header1, header2):
        if v.lower() in {"nan", ""}:
            new_cols.append(h)
        elif h.lower() in {"nan", ""}:
            new_cols.append(v)
        else:
            new_cols.append(f"{h}_{v}")

    df = df.iloc[2:].reset_index(drop=True)
    df.columns = new_cols
    return df


def tag_all_cases(root_dir):
    header_counts = load_header_counts()

    for case_folder in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_folder)
        if not (os.path.isdir(case_path) and case_folder.startswith("wiki")):
            continue

        source_csv_path = os.path.join(case_path, "source.csv")
        output_json_path = os.path.join(case_path, "source_tags.json")

        if not os.path.exists(source_csv_path):
            print(f"Skipping {case_folder}: source.csv not found.")
            continue

        try:
            print(f"Processing tags for: {case_folder}...")
            wiki_id = case_folder.replace("wiki_", "")

            if header_counts.get(wiki_id, 1) == 2:
                raw_df = pd.read_csv(source_csv_path, header=None, dtype=str)
                df = merge_two_headers(raw_df)
            else:
                df = pd.read_csv(source_csv_path, dtype=str)

            tagging_result = call_llm_tagging(df, wiki_folder=case_folder)

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(tagging_result, f, indent=4)

            print(f"Successfully saved tags to {output_json_path}")

        except Exception as e:
            print(f"Error processing {case_folder}: {e}")


if __name__ == "__main__":
    tag_all_cases(INPUT_DIR)
