from __future__ import annotations

import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterator


csv.field_size_limit(sys.maxsize)

_IDX_RE = re.compile(r'"idx"\s*:\s*(\d+)')


class _ZipMemberText(io.TextIOWrapper):
    def __init__(self, zip_path: Path, member: str):
        self._zf = zipfile.ZipFile(zip_path)
        try:
            raw = self._zf.open(member)
        except BaseException:
            self._zf.close()
            raise
        super().__init__(raw, encoding="utf-8", newline="")

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._zf.close()


def _open_member(wiki_dir: Path, filename: str):
    direct = wiki_dir / filename
    if direct.is_file():
        return open(direct, "r", encoding="utf-8", newline="")

    zip_path = wiki_dir / f"{wiki_dir.name}.zip"
    if zip_path.is_file():
        member = f"{wiki_dir.name}/{filename}"
        return _ZipMemberText(zip_path, member)

    raise FileNotFoundError(
        f"{filename} not found for wiki at {wiki_dir} (no raw file, no zip)"
    )


def load_multi_header(wiki_dir: Path) -> bool:
    with _open_member(wiki_dir, "source_tags.json") as f:
        tags = json.load(f)
    return bool(tags.get("multi_header", False))


def load_source_data_rows(wiki_dir: Path, *, multi_header: bool) -> list[list[str]]:
    with _open_member(wiki_dir, "source.csv") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    data_rows = rows[1:]
    if multi_header and data_rows:
        data_rows = data_rows[1:]
    return data_rows


def iter_output_data_rows(
    wiki_dir: Path, wanted_idxs: set[int]
) -> Iterator[tuple[int, list[list[str]]]]:
    if not wanted_idxs:
        return
    remaining = set(wanted_idxs)
    with _open_member(wiki_dir, "outputs.jsonl") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            m = _IDX_RE.search(line)
            if not m:
                continue
            idx = int(m.group(1))
            if idx not in remaining:
                continue
            rec = json.loads(line)
            csv_text = rec["table"]["csv"]
            out_rows = list(csv.reader(io.StringIO(csv_text)))
            data_rows = out_rows[1:] if out_rows else []
            remaining.discard(idx)
            yield idx, data_rows


def tables_identical(input_rows: list[list[str]], output_rows: list[list[str]]) -> bool:
    return input_rows == output_rows


def check_identical_for_wiki(
    wiki_dir: Path,
    idxs: set[int],
    input_rows: list[list[str]],
) -> dict[int, bool]:
    result: dict[int, bool] = {}
    for idx, output_rows in iter_output_data_rows(wiki_dir, set(idxs)):
        result[idx] = tables_identical(input_rows, output_rows)
    return result


