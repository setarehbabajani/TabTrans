from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from .array_config import structural_check_cache_path
from .llm_structural_check import (
    LOCAL_SERVED_MODEL,
    MODEL_SHORT_NAME,
    N_HEAD_ROWS,
    SYSTEM_PROMPT,
    build_table_preview,
    build_user_prompt,
    call_model,
    model_output_path,
)
from .llm_groupby_check import (
    SYSTEM_PROMPT as GROUPBY_SYSTEM_PROMPT,
    build_groupby_prompt,
    parse_groupby_response,
)

CACHE_FILENAME = "structural_check_cache.json"

PHASE1 = "phase1"
GROUPBY = "groupby"

GROUPBY_COL_PREFIX = "groupby_col:"

EMPTY_FLAGS = {
    "transpose_needed": None,
    "column_flags": {},
    "multi_header_wrapped": False,
}


def _entry_response(entry: dict | None) -> dict | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("error"):
        return None
    nested = entry.get("response")
    if isinstance(nested, dict):
        return nested
    if "columns" in entry or "table" in entry:
        return entry
    return None


class _CacheStore:
    _instances: dict[str, "_CacheStore"] = {}

    def __init__(self, path: Path, autosave_every: int = 1):
        self.path = Path(path)
        self.autosave_every = autosave_every
        self.entries: dict[str, dict] = {}
        self._unsaved = 0
        self.loaded_count = 0
        self._load()

    @classmethod
    def get(cls, path, autosave_every: int = 1) -> "_CacheStore":
        key = str(Path(path).resolve())
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls(Path(path), autosave_every=autosave_every)
            cls._instances[key] = inst
        return inst

    def _load(self):
        if not self.path.is_file():
            return
        try:
            t0 = time.perf_counter()
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.entries = data
            self.loaded_count = len(self.entries)
            print(
                f"Loaded structural check cache with {self.loaded_count} "
                f"entries from {self.path} ({time.perf_counter() - t0:.1f}s)"
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to load structural check cache: {exc}")

    def get_entry(self, key: str):
        return self.entries.get(key)

    def put(self, key: str, value: dict):
        self.entries[key] = value
        self._unsaved += 1
        if self.autosave_every and self._unsaved >= self.autosave_every:
            self.save()

    def save(self):
        if not self._unsaved and self.path.is_file():
            return
        if not self.entries and not self._unsaved:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".sc_", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp, self.path)
            print(
                f"Saved structural check cache with {len(self.entries)} "
                f"entries to {self.path}"
            )
            self._unsaved = 0
        except OSError as exc:
            print(f"Failed to save structural check cache: {exc}")


def parse_response(response: dict) -> dict:
    transpose_needed = None
    table = (response or {}).get("table")
    if isinstance(table, dict):
        transpose = table.get("transpose")
        if isinstance(transpose, dict) and "needed" in transpose:
            transpose_needed = bool(transpose["needed"])

    column_flags: dict[str, dict] = {}
    columns = (response or {}).get("columns")
    if isinstance(columns, dict):
        for col_name, info in columns.items():
            if not isinstance(info, dict):
                continue
            entry = {"pivot": None, "unpivot": None, "groupby": None}
            for kind in ("pivot", "unpivot", "groupby"):
                block = info.get(kind)
                if isinstance(block, dict) and "applicable" in block:
                    entry[kind] = bool(block["applicable"])
            column_flags[str(col_name)] = entry

    return {
        "transpose_needed": transpose_needed,
        "column_flags": column_flags,
        "multi_header_wrapped": False,
    }


def is_empty(flags: dict | None) -> bool:
    if not flags:
        return True
    return not flags.get("column_flags") and flags.get("transpose_needed") is None


def columns_match(flags: dict, df: pd.DataFrame) -> bool:
    flagged = set((flags or {}).get("column_flags") or {})
    if not flagged:
        return False
    live = {str(c) for c in df.columns}
    return bool(flagged & live)


class StructuralCheckProvider:
    def __init__(
        self,
        root,
        *,
        kind: str = PHASE1,
        verbose: bool = True,
        autosave_every: int = 1,
    ):
        self.root = Path(root)
        self.kind = kind
        self.verbose = verbose

        if kind == GROUPBY:
            self._system_prompt = GROUPBY_SYSTEM_PROMPT
            self._build_prompt = build_groupby_prompt
            self._parse = parse_groupby_response
        else:
            self._system_prompt = SYSTEM_PROMPT
            self._build_prompt = build_user_prompt
            self._parse = parse_response

        self.store = _CacheStore.get(
            structural_check_cache_path(),
            autosave_every=autosave_every,
        )

    @staticmethod
    def preview_of(df: pd.DataFrame):
        return build_table_preview(df.head(N_HEAD_ROWS))

    def cache_key(self, columns) -> str:
        payload = json.dumps([str(c) for c in columns], ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{self.kind}:{digest}"

    def save(self):
        self.store.save()

    def flags_for(self, df: pd.DataFrame, table_name: str = "") -> dict | None:
        if df is None or df.empty or len(df.columns) == 0:
            return None

        columns = [str(c) for c in df.columns]
        key = self.cache_key(columns)

        entry = self.store.get_entry(key)
        if entry is not None:
            if entry.get("error"):
                return None
            response = _entry_response(entry)
            if response is None:
                return None
            return self._parse(response)

        preview, _ = self.preview_of(df)
        prompt = self._build_prompt(table_name or "table", preview)
        try:
            response = call_model(self._system_prompt, prompt)
        except Exception as exc:
            if self.verbose:
                print(
                    f"[{self.kind.upper()}] check failed for "
                    f"{table_name!r}: {exc}"
                )
            self.store.put(key, {"error": str(exc)})
            return None

        self.store.put(key, response)
        if self.verbose:
            print(
                f"[{self.kind.upper()}] checked {table_name!r} "
                f"({len(columns)} columns)"
            )
        return self._parse(response)

    def flags_for_source(
        self, folder_path, df: pd.DataFrame, existing: dict | None = None
    ) -> dict | None:
        if self.kind != PHASE1:
            raise ValueError("flags_for_source is only valid for the phase1 check")

        folder = Path(folder_path)
        name = folder.name

        if not is_empty(existing):
            if columns_match(existing, df):
                return existing
            if self.verbose:
                print(
                    f"[PHASE1] {name}: existing check does not match the live "
                    "columns, recomputing"
                )
        elif self.verbose:
            print(f"[PHASE1] {name}: no usable structural check on disk")

        flags = self.flags_for(df, table_name=name)
        if flags is None:
            return None

        try:
            entry = self.store.get_entry(
                self.cache_key([str(c) for c in df.columns])
            ) or {}
            response = _entry_response(entry) or {}
            out = model_output_path(folder)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "table": name,
                        "model": {
                            "short_name": MODEL_SHORT_NAME,
                            "model_id": LOCAL_SERVED_MODEL,
                        },
                        "columns": [str(c) for c in df.columns],
                        "n_rows_sampled": min(N_HEAD_ROWS, len(df)),
                        "multi_header_wrapped": False,
                        "status": "ok",
                        "response": response,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            if self.verbose:
                print(f"[PHASE1] wrote {out.name}")
        except Exception as exc:
            if self.verbose:
                print(f"[PHASE1] could not mirror check into {name}: {exc}")

        return flags

    @staticmethod
    def _col_key(column) -> str:
        return f"{GROUPBY_COL_PREFIX}{column}"

    def groupby_column_verdict(self, column) -> bool | None:
        entry = self.store.get_entry(self._col_key(column))
        if not isinstance(entry, dict):
            return None
        if "groupby" not in entry:
            return None
        return bool(entry.get("groupby"))

    def _remember_columns(self, flags: dict, table_name: str = ""):
        for col, info in (flags.get("column_flags") or {}).items():
            if not isinstance(info, dict):
                continue
            verdict = info.get("groupby")
            if verdict is None:
                continue
            key = self._col_key(col)
            if self.store.get_entry(key) is None:
                self.store.put(key, {"groupby": bool(verdict)})

    def groupby_allowed(self, df, column, table_name: str = "") -> bool | None:
        cached = self.groupby_column_verdict(column)
        if cached is not None:
            return cached

        flags = self.flags_for(df, table_name=table_name)
        if flags is None:
            return None

        self._remember_columns(flags, table_name=table_name)

        info = (flags.get("column_flags") or {}).get(str(column))
        if not isinstance(info, dict):
            return None
        return info.get("groupby")
