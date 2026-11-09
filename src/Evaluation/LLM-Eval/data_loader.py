import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

LAYOUTS = ("case", "manifest")


@dataclass
class CaseRef:
    case_id: str
    dataset: str
    in_path: Path
    out_path: Path
    track: str = ""
    composition_type: str = ""
    keys: set = field(default_factory=set)
    extra: dict = field(default_factory=dict)


@dataclass
class Case:
    case_id: str
    dataset: str
    track: str
    composition_type: str
    input_df: pd.DataFrame
    gold_df: pd.DataFrame
    has_orig_index: bool
    extra_metadata: dict


def list_cases(layout: str, data_dir, dataset: str = None) -> list:
    data_dir = Path(data_dir)
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if layout == "case":
        return _list_case_layout(data_dir, dataset)
    return _list_manifest_layout(data_dir, dataset)


def _manifest_ids(d: Path) -> dict:
    man = d / "manifest.csv"
    if not man.exists():
        return {}
    try:
        df = pd.read_csv(man, dtype=str, keep_default_na=False)
    except Exception:
        return {}
    if "example_id" not in df.columns:
        return {}
    out = {}
    for i, r in df.iterrows():
        m = re.fullmatch(r"case_(\d+)_in\.csv", Path(r.get("input_csv", "") or "").name)
        out[int(m.group(1)) if m else int(i)] = str(r["example_id"])
    return out


def _list_case_layout(d: Path, dataset):
    refs = []
    ids = _manifest_ids(d)
    for p in d.glob("case_*_in.csv"):
        m = re.fullmatch(r"case_(\d+)_in\.csv", p.name)
        if not m:
            continue
        n = m.group(1)
        out = d / f"case_{n}_out.csv"
        if not out.exists():
            print(f"WARNING: {p.name} has no {out.name}; skipped")
            continue
        comp, ds, extra = d.name, dataset or d.name, {}
        pj = d / f"case_{n}_pipeline.json"
        if pj.exists():
            try:
                meta = json.loads(pj.read_text())
                comp = str(meta.get("composition_type", "") or comp)
                ds = dataset or str(meta.get("dataset", "") or d.name)
                extra = {k: v for k, v in meta.items() if k not in ("composition_type", "dataset")}
            except json.JSONDecodeError as e:
                print(f"WARNING: bad JSON in {pj.name}: {e}")
        refs.append(CaseRef(
            case_id=f"case_{n}", dataset=ds, in_path=p, out_path=out,
            track=d.name, composition_type=comp,
            keys={f"case_{n}", f"{d.name}_{n}", f"{ds}_{n}"} | ({ids[int(n)]} if int(n) in ids else set()),
            extra={**extra, "example_id": ids.get(int(n), "")},
        ))
    refs.sort(key=lambda r: int(r.case_id.split("_")[1]))
    return refs


def _list_manifest_layout(d: Path, dataset):
    man = d / "manifest.csv" if d.is_dir() else d
    if d.is_dir() and not man.exists() and (d / "meta.csv").exists():
        man = d / "meta.csv"
    base = man.parent
    df = pd.read_csv(man, dtype=str, keep_default_na=False)
    df = df.rename(columns={"source_csv": "input_csv", "target_csv": "output_csv"})
    need = {"example_id", "input_csv", "output_csv"}
    if not need <= set(df.columns):
        raise ValueError(f"{man} needs columns {sorted(need)}; has {list(df.columns)}")
    refs = []
    for _, r in df.iterrows():
        eid = str(r["example_id"])
        refs.append(CaseRef(
            case_id=eid, dataset=dataset or base.name,
            in_path=base / r["input_csv"], out_path=base / r["output_csv"],
            track=str(r.get("track", "") or ""), composition_type=base.name,
            keys={eid},
        ))
    return refs


def load_case(ref: CaseRef) -> Case:
    input_df = pd.read_csv(ref.in_path)
    gold_df = pd.read_csv(ref.out_path)
    return Case(
        case_id=ref.case_id, dataset=ref.dataset, track=ref.track,
        composition_type=ref.composition_type,
        input_df=input_df, gold_df=gold_df,
        has_orig_index="orig_index" in gold_df.columns,
        extra_metadata=ref.extra,
    )
