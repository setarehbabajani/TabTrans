import os
import json
import random
import pandas as pd
from collections import deque
from itertools import combinations
import hashlib
from pathlib import Path
import traceback
import re

from src.tagging.call_llm import WikiTaggingFailed, call_llm_tagging
from .state import TableState
from .transformation_loader import load_transformations
import src.transformations.structural.registry as structural_registry
from src.transformations.structural.wrap_multi_header import WrapMultiHeader
from .array_config import project_root, tag_cache_path
from .output_store import OutputStore
from .structural_planner import (
    plan_phase1_candidates,
    build_step_record,
)
from .structural_check_provider import StructuralCheckProvider, GROUPBY


cleanup_names = {
    "drop_columns",
    "drop_duplicates",
    "drop_rows"
}

STRUCTURAL_CHECK_MANIFEST = "structural_transform_check.json"


def _structural_check_model_files(folder: Path) -> list:
    model_files = []
    manifest_path = folder / STRUCTURAL_CHECK_MANIFEST
    if manifest_path.is_file():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            for info in (manifest.get("models") or {}).values():
                if not isinstance(info, dict) or info.get("status") != "ok":
                    continue
                name = info.get("file")
                if name:
                    path = folder / name
                    if path.is_file():
                        model_files.append(path)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    if not model_files:
        model_files = sorted(
            p
            for p in folder.glob("structural_transform_check.*.json")
            if p.is_file()
        )
    return model_files


def load_structural_check(folder_path: str) -> dict:
    folder = Path(folder_path)
    empty = {
        "transpose_needed": None,
        "column_flags": {},
        "multi_header_wrapped": False,
    }

    for path in _structural_check_model_files(folder):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if payload.get("status") not in (None, "ok"):
            continue

        response = payload.get("response")
        if not isinstance(response, dict):
            continue

        transpose_needed = None
        table = response.get("table")
        if isinstance(table, dict):
            transpose = table.get("transpose")
            if isinstance(transpose, dict) and "needed" in transpose:
                transpose_needed = bool(transpose["needed"])

        column_flags = {}
        columns = response.get("columns")
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
            "multi_header_wrapped": bool(
                payload.get("multi_header_wrapped", False)
            ),
        }

    return empty


class PipelineEngine:
    def __init__(self, *, skip_io: bool = False):
        self.skip_io = skip_io
        self._save_count = 0

        self.structural = structural_registry.get_all()

        self.by_name = {t.name: t for t in self.structural}

        self._structural_provider = StructuralCheckProvider(project_root())
        self._groupby_provider = StructuralCheckProvider(project_root(), kind=GROUPBY)

        self.atomic = load_transformations()

        self.cleanup = [t for t in self.structural if t.name in cleanup_names]

        self.tag_cache = {}
        self.load_tag_cache()

    def hash_df(self, df):
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=False).values
        ).hexdigest()

    def get_col_hash(self, series):
        return hashlib.sha256(
            pd.util.hash_pandas_object(series, index=False).values
        ).hexdigest()

    def load_tag_cache(self, path=None):
        path = path or str(tag_cache_path())
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.tag_cache = json.load(f)
                print(f"Loaded tag cache with {len(self.tag_cache)} entries from {path}")
            except Exception as e:
                print("Failed to load tag cache:", e)

    def save_tag_cache(self, path=None):
        path = path or str(tag_cache_path())
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.tag_cache, f, indent=2)
            print(f"Saved tag cache with {len(self.tag_cache)} entries to {path}")
        except Exception as e:
            print("Failed to save tag cache:", e)

    def _resolve_missing_tags(self, df, tag_list, param_list):
        n = len(df.columns)
        tag_list = list(tag_list) if tag_list is not None else [None] * n
        param_list = list(param_list) if param_list is not None else [{}] * n

        if len(tag_list) < n:
            tag_list.extend([None] * (n - len(tag_list)))
        if len(param_list) < n:
            param_list.extend([{}] * (n - len(param_list)))

        need_llm_indices = []

        for i in range(n):
            if tag_list[i] is not None:
                continue

            c_hash = self.get_col_hash(df.iloc[:, i])

            if c_hash in self.tag_cache:
                cached = self.tag_cache[c_hash]
                tag_list[i] = cached[0]
                param_list[i] = cached[1]
            else:
                need_llm_indices.append(i)

        if not need_llm_indices:
            return tag_list, param_list

        subset = df.iloc[:, need_llm_indices].copy()
        subset.columns = [str(df.columns[i]) for i in need_llm_indices]

        print(
            f"LLM tagging {len(need_llm_indices)} column(s) "
            f"(skipping {n - len(need_llm_indices)} already tagged/cached)"
        )

        llm_output = call_llm_tagging(
            subset,
            wiki_folder=getattr(self, "_current_wiki", None),
        )

        if llm_output is None:
            return None, None

        llm_map = {
            col["column_name"]: (
                col.get("tag", "STRING"),
                col.get("parameters", {}) or {},
            )
            for col in llm_output["columns"]
        }

        for i in need_llm_indices:
            col_name = str(df.columns[i])
            tag, param = llm_map.get(col_name, ("STRING", {}))
            tag_list[i] = tag
            param_list[i] = param
            self.tag_cache[self.get_col_hash(df.iloc[:, i])] = (tag, param)

        self.save_tag_cache()
        return tag_list, param_list

    def _new_output_column_names(self, state, cols, new_cols, temp_df):
        if len(cols) == len(new_cols):
            return [temp_df.columns[c] for c in cols]

        if len(cols) == 1 and len(new_cols) > 1:
            original_name = state.df.columns[cols[0]]
            return [f"{original_name}_{i}" for i in range(len(new_cols))]

        if len(cols) > 1 and len(new_cols) == 1:
            cols_sorted = sorted(cols)
            names = [state.df.columns[c] for c in cols_sorted]
            return ["__MERGED__".join(names)]

        return [f"col_{i}" for i in range(len(new_cols))]

    def build_tags(self, df, tag_list, param_list, multi_header):
        return {
            "multi_header": multi_header,
            "columns": [
                {
                    "column_name": col,
                    "tag": None if not tag_list or tag_list[i] is None else tag_list[i],
                    "parameters": {} if not param_list or param_list[i] is None else param_list[i]
                }
                for i, col in enumerate(df.columns)
            ]
        }

    def get_structural_candidates(self, state, transforms):
        candidates = []
        df = state.df

        tag_list = [c["tag"] for c in state.tags["columns"]]
        param_list = [c["parameters"] for c in state.tags["columns"]]

        for t in transforms:
            already_used = any(
                step.get("phase") == "phase3_structural"
                and step.get("transformation") == t.name
                for step in state.steps
            )

            if already_used:
                print(f"Skipping repeated phase3 transformation: {t.name}")
                continue

            try:
                if t.check(df, tag_list, param_list):
                    candidates.append((t, None))
            except Exception:
                continue

        return candidates

    def protected_cell_columns(self, state):
        protected = set()
        for step in state.steps or []:
            for col in step.get("protected") or []:
                protected.add(str(col))
        return protected

    def get_cell_candidates(self, state):
        df = state.df
        tags = state.tags

        candidates = []
        protected = self.protected_cell_columns(state)
        if protected:
            print(f"Phase 2: protected columns (no cell-level ops): {sorted(protected)}")

        param_list = [c["parameters"] for c in tags["columns"]]

        n_cols = len(df.columns)

        column_series = [df.iloc[:, i] for i in range(n_cols)]

        protected_indices = {
            i for i in range(n_cols) if str(df.columns[i]) in protected
        }

        for t in self.atomic:
            if t.input_arity == 1:
                for col_idx in range(n_cols):
                    if col_idx in protected_indices:
                        continue

                    col_series = column_series[col_idx]
                    col_tag = tags["columns"][col_idx]["tag"]
                    col_params = tags["columns"][col_idx]["parameters"]

                    history = state.column_history.get(state.df.columns[col_idx], [])

                    already_applied = any(
                        h.get("transformation") == t.name
                        for h in history
                    )

                    if already_applied:
                        continue

                    try:
                        if t.check([col_series], [col_tag], col_params):
                            candidates.append((t, [col_idx]))
                    except Exception:
                        continue

            elif t.input_arity == 2:
                for cols in combinations(range(n_cols), 2):
                    if any(c in protected_indices for c in cols):
                        continue

                    col_series_list = [column_series[c] for c in cols]
                    col_tags = [tags["columns"][c]["tag"] for c in cols]
                    col_params = [param_list[c] for c in cols]

                    try:
                        if t.check(col_series_list, col_tags, col_params):
                            candidates.append((t, list(cols)))
                    except Exception:
                        continue

            elif t.input_arity == "n":
                max_group = min(3, n_cols)

                for r in range(1, max_group + 1):
                    for cols in combinations(range(n_cols), r):
                        cols = list(cols)

                        if any(c in protected_indices for c in cols):
                            continue

                        col_series_list = [column_series[c] for c in cols]
                        col_tags = [tags["columns"][c]["tag"] for c in cols]
                        col_params = [param_list[c] for c in cols]

                        try:
                            if t.check(col_series_list, col_tags, col_params):
                                candidates.append((t, cols))
                        except Exception:
                            continue

        return candidates

    def creates_parameter_loop(self, state, t, cols, params):
        if t.name == "concat_columns":
            split_group_ids = set()
            for c in cols:
                col_name = state.df.columns[c]

                gid = state.split_groups.get(col_name)

                if gid is not None:
                    split_group_ids.add(gid)

            if len(split_group_ids) == 1:
                gid = next(iter(split_group_ids))

                cols_in_group = [
                    name
                    for name, g in state.split_groups.items()
                    if g == gid
                ]

                selected_names = [
                    state.df.columns[c]
                    for c in cols
                ]

                if set(cols_in_group) == set(selected_names):
                    previous_split = None
                    for c in cols:
                        col_name = state.df.columns[c]
                        hist = state.column_history.get(col_name, [])
                        for h in reversed(hist):
                            if (
                                isinstance(h, dict)
                                and h.get("transformation") == "split_column"
                            ):
                                previous_split = h
                                break

                    if previous_split:
                        split_params = previous_split.get("params", {})

                        split_delim = split_params.get("delimiter")

                        param_list = params if isinstance(params, list) else [params]
                        concat_delim = next(
                            (
                                p.get("delimiter")
                                for p in param_list
                                if isinstance(p, dict) and p.get("delimiter")
                            ),
                            None,
                        )

                        if split_delim == concat_delim:
                            return True

        if t.name == "split_column":
            col_name = state.df.columns[cols[0]]
            hist = state.column_history.get(col_name, [])

            for h in reversed(hist):
                if isinstance(h, dict) and h.get("transformation") == "concat_columns":
                    concat_params = h.get("params", {})

                    concat_delim = concat_params.get("delimiter")
                    split_delim = params.get("delimiter")

                    if concat_delim == split_delim:
                        return True

        return False

    def apply_structural(
        self,
        state,
        t,
        visited_hashes,
        phase_name,
        apply_kwargs=None,
        targets=None,
    ):
        apply_kwargs = apply_kwargs or {}
        try:
            print("Applying:", t.name, apply_kwargs or "")
            tag_list = [
                c.get("tag")
                for c in state.tags["columns"]
            ]

            param_list = [
                c.get("parameters")
                for c in state.tags["columns"]
            ]

            new_df, new_tag_list, new_param_list = t.apply(
                state.df.copy(),
                tag_list,
                param_list,
                **apply_kwargs
            )

            if new_df is None:
                return None

            df_hash = self.hash_df(new_df)

            if df_hash in visited_hashes:
                print("Duplicate structural table")
                return None

            if new_tag_list is None or any(tag is None for tag in new_tag_list):
                if new_tag_list is None:
                    new_tag_list = [None] * len(new_df.columns)

                if new_param_list is None:
                    new_param_list = [{}] * len(new_df.columns)

                resolved = self._resolve_missing_tags(
                    new_df,
                    new_tag_list,
                    new_param_list,
                )

                new_tag_list, new_param_list = resolved

                if new_tag_list is None:
                    return None

            new_tags = self.build_tags(
                new_df,
                new_tag_list,
                new_param_list,
                state.tags.get("multi_header", False)
            )

            new_state = TableState(
                new_df,
                new_tags,
                state.steps + [
                    build_step_record(
                        phase_name,
                        t,
                        apply_kwargs,
                        targets,
                        created_diff=[
                            c for c in new_df.columns
                            if str(c) not in {str(x) for x in state.df.columns}
                        ],
                    )
                ],
                state.structural_count + 1,
                {},
                state.structural_used.copy(),
                {},
                split_groups=state.split_groups.copy(),
                merged_columns=state.merged_columns.copy()
            )
            if hasattr(state, "phase2_id"):
                new_state.phase2_id = state.phase2_id

            return new_state

        except WikiTaggingFailed:
            raise
        except Exception as e:
            print("Structural apply error:", e)
            return None

    def apply_cell_transformation(self, state, t, cols, visited_hashes):
        try:
            original_columns = list(state.df.columns)

            col_series = [state.df[state.df.columns[c]] for c in cols]
            col_tags = [state.tags["columns"][c]["tag"] for c in cols]
            col_params = [state.tags["columns"][c]["parameters"] for c in cols]

            if t.input_arity == 1:
                params = (
                    col_params[0]
                    if col_params and isinstance(col_params[0], dict)
                    else {}
                )
            else:
                params = [
                    p if isinstance(p, dict) else {}
                    for p in col_params
                ]

            if self.creates_parameter_loop(state, t, cols, params):
                print("Parameter loop prevented")
                return None

            if hasattr(t, "last_used_params"):
                t.last_used_params = None
            if hasattr(t, "output_name_hints"):
                t.output_name_hints = None

            if not t.check(col_series, col_tags, params):
                return None

            new_cols, new_tags, new_params = t.apply(
                col_series,
                col_tags,
                params
            )
            if new_cols is None:
                return None
            if isinstance(new_cols, list) and len(new_cols) == len(cols):
                if all(
                    pd.Series(old).reset_index(drop=True).equals(
                        pd.Series(new).reset_index(drop=True)
                    )
                    for old, new in zip(col_series, new_cols)
                ):
                    return None

            if isinstance(new_params, dict):
                new_params = [new_params]

            if new_params is None:
                new_params = [{} for _ in range(len(new_cols))]

            if new_tags is None or any(tag is None for tag in new_tags):
                temp_df = state.df.copy()

                if len(cols) == len(new_cols):
                    for i, c in enumerate(cols):
                        temp_df[temp_df.columns[c]] = new_cols[i]

                elif len(cols) == 1 and len(new_cols) > 1:
                    c = cols[0]
                    col_name = temp_df.columns[c]

                    split_df = pd.DataFrame(
                        {
                            f"{col_name}_{i}": new_cols[i]
                            for i in range(len(new_cols))
                        },
                        index=temp_df.index,
                    )
                    left = temp_df.iloc[:, :c]
                    right = temp_df.iloc[:, c + 1 :]
                    temp_df = pd.concat([left, split_df, right], axis=1)

                elif len(cols) > 1 and len(new_cols) == 1:
                    cols_sorted = sorted(cols)
                    col_names = [temp_df.columns[c] for c in cols_sorted]

                    temp_df = temp_df.drop(columns=col_names)
                    temp_df.insert(cols_sorted[0], "__MERGED__".join(col_names), new_cols[0])

                df_hash = self.hash_df(temp_df)

                if df_hash in visited_hashes:
                    return None

                if new_tags is None:
                    new_tags = [None] * len(new_cols)

                col_names = self._new_output_column_names(
                    state, cols, new_cols, temp_df
                )
                untagged_df = pd.DataFrame(
                    {
                        col_names[i]: new_cols[i]
                        for i in range(len(new_cols))
                        if new_tags[i] is None
                    }
                )

                if not untagged_df.empty:
                    resolved = self._resolve_missing_tags(
                        untagged_df,
                        [None] * len(untagged_df.columns),
                        [{}] * len(untagged_df.columns),
                    )

                    resolved_tags, resolved_params = resolved

                    if resolved_tags is None:
                        return None

                    for j, name in enumerate(untagged_df.columns):
                        i = col_names.index(name)
                        new_tags[i] = resolved_tags[j]
                        new_params[i] = resolved_params[j]

            if not isinstance(new_cols, list):
                print(f"{t.name} returned invalid columns")
                return None

            if not isinstance(new_params, list):
                new_params = [new_params] * len(new_cols)

            new_state = TableState(
                state.df.copy(),
                json.loads(json.dumps(state.tags)),
                state.steps + [{
                    "phase": "phase2_cell_level",
                    "transformation": t.name,
                    "columns": [original_columns[c] for c in cols],
                    "params": (
                        getattr(t, "last_used_params", None)
                        if isinstance(getattr(t, "last_used_params", None), dict)
                        else (params if isinstance(params, dict) else {})
                    )
                }],
                state.structural_count,
                state.column_counts.copy(),
                state.structural_used.copy(),
                state.column_history.copy(),
                state.split_groups.copy(),
                state.merged_columns.copy()
            )

            if len(new_cols) == len(cols):
                for i, c in enumerate(cols):
                    col_name = original_columns[c]

                    new_state.df[col_name] = new_cols[i]

                    new_state.tags["columns"][c]["tag"] = new_tags[i]
                    new_state.tags["columns"][c]["parameters"] = new_params[i]

                    if c not in new_state.column_history:
                        new_state.column_history[col_name] = []

                    new_state.column_history[col_name].append({
                        "transformation": t.name,
                        "params": params
                    })

                return new_state

            if len(cols) == 1 and len(new_cols) > 1:
                c = cols[0]
                col_name = new_state.df.columns[c]

                hints = getattr(t, "output_name_hints", None)
                if isinstance(hints, list) and len(hints) == len(new_cols):
                    suffixes = [str(h) for h in hints]
                else:
                    suffixes = [str(i) for i in range(len(new_cols))]

                new_state.df = new_state.df.drop(columns=[col_name])
                new_state.tags["columns"].pop(c)
                new_state.split_groups.pop(col_name, None)
                group_id = random.randint(1, 10**9)

                split_df = pd.DataFrame(
                    {
                        f"{col_name}_{suffixes[i]}": new_cols[i]
                        for i in range(len(new_cols))
                    },
                    index=new_state.df.index,
                )
                left = new_state.df.iloc[:, :c]
                right = new_state.df.iloc[:, c:]
                new_state.df = pd.concat([left, split_df, right], axis=1)

                for i in range(len(new_cols)):
                    new_col_name = f"{col_name}_{suffixes[i]}"
                    insert_pos = c + i

                    new_state.tags["columns"].insert(insert_pos, {
                        "column_name": new_col_name,
                        "tag": new_tags[i],
                        "parameters": new_params[i]
                    })
                    new_state.column_history[new_col_name] = (
                        state.column_history.get(col_name, []).copy()
                    )
                    new_state.column_history[new_col_name].append({
                        "transformation": t.name,
                        "params": new_params[i]
                    })
                    new_state.split_groups[new_col_name] = group_id

                return new_state

            if len(cols) > 1 and len(new_cols) == 1:
                cols = sorted(cols)
                first_col = cols[0]

                col_names = [original_columns[c] for c in cols]

                new_state.df = new_state.df.drop(columns=col_names)

                for c in sorted(cols, reverse=True):
                    new_state.tags["columns"].pop(c)

                new_col_name = "__MERGED__".join(col_names)
                group_ids = set()
                for name in col_names:
                    new_state.split_groups.pop(name, None)
                    gid = state.split_groups.get(name)
                    if gid is not None:
                        group_ids.add(gid)

                if len(group_ids) == 1:
                    new_state.split_groups[new_col_name] = next(iter(group_ids))
                else:
                    new_state.split_groups[new_col_name] = None

                new_state.df.insert(first_col, new_col_name, new_cols[0])

                new_state.tags["columns"].insert(first_col, {
                    "column_name": new_col_name,
                    "tag": new_tags[0],
                    "parameters": new_params[0]
                })
                merged_history = []

                for c in cols:
                    name = original_columns[c]
                    merged_history.extend(
                        state.column_history.get(name, [])
                    )
                merged_history.append({
                    "transformation": t.name,
                    "params": new_params[0]
                })
                new_state.column_history[new_col_name] = merged_history

                return new_state

            if len(cols) > 1 and len(new_cols) > 1:
                cols_sorted = sorted(cols)
                col_names = [original_columns[c] for c in cols_sorted]
                first_col = cols_sorted[0]

                new_state.df = new_state.df.drop(columns=col_names)

                for c in sorted(cols_sorted, reverse=True):
                    new_state.tags["columns"].pop(c)

                merged_name = "__MERGED__".join(col_names)

                hints = getattr(t, "output_name_hints", None)
                if isinstance(hints, list) and len(hints) == len(new_cols):
                    suffixes = [str(h) for h in hints]
                else:
                    suffixes = [str(i) for i in range(len(new_cols))]

                out_df = pd.DataFrame(
                    {
                        f"{merged_name}_{suffixes[i]}": new_cols[i]
                        for i in range(len(new_cols))
                    },
                    index=new_state.df.index,
                )

                left = new_state.df.iloc[:, :first_col]
                right = new_state.df.iloc[:, first_col:]
                new_state.df = pd.concat([left, out_df, right], axis=1)

                for name in col_names:
                    new_state.split_groups.pop(name, None)

                group_id = random.randint(1, 10**9)

                merged_history = []
                for name in col_names:
                    merged_history.extend(
                        state.column_history.get(name, [])
                    )

                for i in range(len(new_cols)):
                    new_col_name = f"{merged_name}_{suffixes[i]}"
                    insert_pos = first_col + i

                    new_state.tags["columns"].insert(insert_pos, {
                        "column_name": new_col_name,
                        "tag": new_tags[i],
                        "parameters": new_params[i]
                    })

                    new_state.column_history[new_col_name] = (
                        merged_history.copy()
                    )
                    new_state.column_history[new_col_name].append({
                        "transformation": t.name,
                        "params": new_params[i]
                    })
                    new_state.split_groups[new_col_name] = group_id

                return new_state

            return None

        except WikiTaggingFailed:
            raise
        except Exception:
            print("Cell-level apply error:")
            traceback.print_exc()
            return None

    def structural_flags_for(self, df, table_name=""):
        return self._structural_provider.flags_for(df, table_name=table_name)


    def groupby_target(self, state, t):
        tag_list = [c.get("tag") for c in state.tags["columns"]]
        group_col = t.select_group_column(state.df, tag_list)
        if group_col is None:
            return None, None
        numeric_col = t.select_numeric_column(state.df, tag_list, group_col)
        if numeric_col is None:
            return None, None
        return group_col, numeric_col

    def groupby_confirmed(self, df, group_col, table_name=""):
        verdict = self._groupby_provider.groupby_allowed(
            df, group_col, table_name=table_name
        )
        if verdict is True:
            return True
        print(
            f"Skipped groupby_aggregate: check says {group_col!r} is not a "
            f"meaningful grouping key (verdict={verdict})"
        )
        return False

    def run_on_folder(self, folder_path):
        wiki_name = os.path.basename(folder_path)
        self._current_wiki = wiki_name
        print(f"Pipeline start: {wiki_name}" + (" [no-io]" if self.skip_io else ""))

        self._save_count = 0
        if self.skip_io:
            self._output_store = None
        else:
            self._output_store = OutputStore(folder_path)
            self._output_store.begin()

        df = pd.read_csv(os.path.join(folder_path, "source.csv"))

        with open(os.path.join(folder_path, "source_tags.json")) as f:
            tags = json.load(f)

        multi_header = tags.get("multi_header", False)

        idx = 0

        # Multi-row header: merging the two header rows before any transformation.
        if multi_header:
            df_copy = df.copy(deep=True)
            df, _, _ = WrapMultiHeader().apply(df_copy, None, None)

            tags["multi_header"] = False

            initial = TableState(
                df,
                tags,
                steps=[{
                    "phase": "phase1_structural",
                    "transformation": "wrap_multi_header",
                    "columns": None
                }],
                structural_count=1,
                column_counts={},
                structural_used={"wrap_multi_header"},
                column_history={},
                split_groups={col: None for col in df.columns},
                merged_columns=set()
            )

            idx += 1

        else:
            initial = TableState(
                df,
                tags,
                steps=[],
                structural_count=0,
                column_counts={},
                structural_used=set(),
                column_history={},
                split_groups={col: None for col in df.columns},
                merged_columns=set()
            )

        visited_hashes = set()
        seen_signatures = set()

        # Structural check for the source table: which columns the LLM marks
        # for pivot / unpivot, and whether the table needs a transpose.
        structural_check = load_structural_check(folder_path)
        structural_check = (
            self._structural_provider.flags_for_source(
                folder_path, df, existing=structural_check
            )
            or structural_check
            or {}
        )
        transpose_needed = structural_check.get("transpose_needed")
        column_flags = structural_check.get("column_flags") or {}
        if transpose_needed is False:
            print(
                "Structural check: transpose.needed=false "
                "(skip transpose as first phase-1 transform)"
            )
        elif transpose_needed is True:
            print("Structural check: transpose.needed=true")
        if column_flags:
            print(
                f"Structural check: loaded flags for {len(column_flags)} columns "
                f"(multi_header_wrapped="
                f"{structural_check.get('multi_header_wrapped')})"
            )

        # Phase 1: structural transformations (pivot, unpivot, transpose,
        # divide_mixed). Each new table gets its own structural check, and its
        # branch keeps going while the check marks something applicable.
        phase1_queue = deque([initial])
        phase1_outputs = [initial]

        flags_by_state = {
            id(initial): {
                "transpose_needed": transpose_needed,
                "column_flags": column_flags,
            }
        }

        while phase1_queue:
            state = phase1_queue.popleft()

            flags = flags_by_state.pop(id(state), None)
            if flags is None:
                continue

            candidates = plan_phase1_candidates(
                state,
                self.by_name,
                flags,
            )

            for t, apply_kwargs, targets in candidates:
                new_state = self.apply_structural(
                    state,
                    t,
                    visited_hashes,
                    "phase1_structural",
                    apply_kwargs=apply_kwargs,
                    targets=targets,
                )

                if new_state is None or new_state.df.equals(state.df):
                    print("phase 1: no new table")
                    continue

                if len(new_state.df) == 0:
                    print("phase 1: empty table")
                    continue

                df_hash = self.hash_df(new_state.df)

                if df_hash in visited_hashes:
                    continue

                visited_hashes.add(df_hash)

                pipeline_sig = tuple(
                    (
                        step["transformation"],
                        tuple(step.get("columns") or ()),
                    )
                    for step in new_state.steps
                )

                if pipeline_sig not in seen_signatures:
                    seen_signatures.add(pipeline_sig)

                    self.save_output(folder_path, idx, new_state)
                    phase1_outputs.append(new_state)

                    idx += 1

                child_flags = self.structural_flags_for(
                    new_state.df,
                    table_name=f"{wiki_name}#{idx}",
                )
                if child_flags is not None:
                    flags_by_state[id(new_state)] = child_flags
                    phase1_queue.append(new_state)

        print("End of phase 1")

        # Phase 2: cell-level transformations, chosen from the column tags,
        # applied breadth-first on every Phase 1 output.
        phase2_outputs = []
        phase2_output_id = 0

        for phase1_state in phase1_outputs:
            queue = deque([phase1_state])

            while queue:
                state = queue.popleft()

                candidates = self.get_cell_candidates(state)

                for t, cols in candidates:
                    base_state = TableState(
                        state.df.copy(deep=True),
                        json.loads(json.dumps(state.tags)),
                        list(state.steps),
                        state.structural_count,
                        state.column_counts.copy(),
                        state.structural_used.copy(),
                        {c: hist.copy() for c, hist in state.column_history.items()},
                        split_groups=state.split_groups.copy(),
                        merged_columns=state.merged_columns.copy()
                    )

                    new_state = self.apply_cell_transformation(
                        base_state,
                        t,
                        cols,
                        visited_hashes
                    )

                    if new_state is None:
                        continue

                    df_hash = self.hash_df(new_state.df)

                    if df_hash in visited_hashes:
                        continue

                    phase2_steps = [
                        step
                        for step in new_state.steps
                        if step.get("phase") == "phase2_cell_level"
                    ]

                    column_sequences = {}
                    for step in phase2_steps:
                        cols = step.get("columns", [])
                        step_roots = set()
                        for col in cols:
                            if re.match(r"^.+_\d+$", col):
                                root_col = col.rsplit("_", 1)[0]
                            else:
                                root_col = col
                            step_roots.add(root_col)
                        for root_col in step_roots:
                            if root_col not in column_sequences:
                                column_sequences[root_col] = []
                            column_sequences[root_col].append((
                                step["transformation"],
                                json.dumps(
                                    step.get("params", {}),
                                    sort_keys=True
                                )
                            ))
                    pipeline_sig = tuple(sorted((col, tuple(seq)) for col, seq in column_sequences.items()))

                    if pipeline_sig not in seen_signatures:
                        seen_signatures.add(pipeline_sig)
                        queue.append(new_state)
                        visited_hashes.add(df_hash)
                        new_state.phase2_id = phase2_output_id
                        phase2_output_id += 1
                        self.save_output(folder_path, idx, new_state)
                        phase2_outputs.append(new_state)
                        idx += 1

        print("End of phase 2")

        # Phase 3: cleanup (drop_columns, drop_duplicates, drop_rows) on every
        # Phase 2 output, then groupby_aggregate and sort_rows.
        cleanup_outputs = []
        for phase2_state in phase2_outputs:
            state = phase2_state
            for t in self.cleanup:
                try:
                    candidates = self.get_structural_candidates(
                        state,
                        [t]
                    )

                    if not candidates:
                        continue

                    new_state = self.apply_structural(
                        state,
                        t,
                        visited_hashes,
                        "phase3_structural"
                    )

                    if (
                        new_state is None
                        or new_state.df.empty
                    ):
                        continue
                    state = new_state
                except WikiTaggingFailed:
                    raise
                except Exception:
                    continue

            df_hash = self.hash_df(state.df)

            if df_hash not in visited_hashes:
                visited_hashes.add(df_hash)
                self.save_output(folder_path, idx, state)
                idx += 1

            cleanup_outputs.append(state)

        print("Cleanup transformations finished")

        # Phase 3, second part: groupby_aggregate (confirmed by the LLM groupby
        # check), then sort_rows.
        groupby_t = self.by_name.get("groupby_aggregate")
        sort_t = self.by_name.get("sort_rows")

        for cleanup_state in cleanup_outputs:

            state = cleanup_state

            if groupby_t is not None:
                try:
                    tag_list = [c.get("tag") for c in state.tags["columns"]]
                    param_list = [c.get("parameters") for c in state.tags["columns"]]

                    if groupby_t.check(state.df, tag_list, param_list):
                        group_col, numeric_col = self.groupby_target(state, groupby_t)

                        if group_col is not None:
                            gb_state = self.apply_structural(
                                state,
                                groupby_t,
                                visited_hashes,
                                "phase3_structural",
                                apply_kwargs={"group_col": group_col},
                                targets=[group_col],
                            )

                            if gb_state is not None and not gb_state.df.empty:
                                gb_hash = self.hash_df(gb_state.df)

                                if gb_hash not in visited_hashes:
                                    if self.groupby_confirmed(
                                        state.df,
                                        group_col,
                                        table_name=f"groupby:{group_col}",
                                    ):
                                        state = gb_state
                except WikiTaggingFailed:
                    raise
                except Exception as exc:
                    print("phase 3 groupby error:", exc)

            if sort_t is not None:
                try:
                    tag_list = [c.get("tag") for c in state.tags["columns"]]
                    param_list = [c.get("parameters") for c in state.tags["columns"]]

                    if sort_t.check(state.df, tag_list, param_list):
                        sorted_state = self.apply_structural(
                            state,
                            sort_t,
                            visited_hashes,
                            "phase3_structural",
                        )
                        if sorted_state is not None and not sorted_state.df.empty:
                            sorted_hash = self.hash_df(sorted_state.df)
                            if sorted_hash not in visited_hashes:
                                state = sorted_state
                            else:
                                print("phase 3 sort_rows: duplicate table, kept unsorted")
                except WikiTaggingFailed:
                    raise
                except Exception as exc:
                    print("phase 3 sort_rows error:", exc)

            if state is cleanup_state:
                continue

            final_hash = self.hash_df(state.df)
            if final_hash in visited_hashes:
                continue

            visited_hashes.add(final_hash)
            self.save_output(folder_path, idx, state)
            idx += 1

        self.save_tag_cache()
        self._structural_provider.save()
        self._groupby_provider.save()
        print("End of phase 3")

        if self.skip_io:
            output_count = self._save_count
        else:
            output_count = self._output_store.count
        print(f"Pipeline end: {wiki_name} ({output_count} outputs)")
        self._output_store = None

        return output_count

    def save_output(self, folder, idx, state):
        if self.skip_io:
            self._save_count += 1
            return
        self._output_store.append(idx, state)
