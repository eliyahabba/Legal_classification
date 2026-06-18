# -*- coding: utf-8 -*-
"""
Build the legal-category validation experiment, and resume files.

Two modes:

  BUILD (default) — create the annotation files for both annotators:
      python build_annotation.py exp9
      python build_annotation.py path/to/classification_results.csv

  RESUME — rebuild one annotator's file from a returned results JSON, with her
  previous answers baked in, opening on the questions she hasn't tagged yet:
      python build_annotation.py resume results_carmit_exp9.json
      python build_annotation.py resume results_carmit_exp9.json path/to/classification_results.csv

Outputs land next to this script. The template (template.html) must sit next to it.
"""
import json
import re
import sys
import pandas as pd
from pathlib import Path

# ---------- paths (independent of where you run from) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE   = SCRIPT_DIR / "template.html"
OUT        = SCRIPT_DIR

def _find_repo_root():
    for start in (SCRIPT_DIR, Path.cwd().resolve()):
        for d in (start, *start.parents):
            if (d / "data" / "experiments").is_dir():
                return d
    return None

REPO_ROOT = _find_repo_root()
EXPERIMENTS_DIR = (REPO_ROOT / "data" / "experiments") if REPO_ROOT else Path("data/experiments")

# ---------- config ----------
PER_CAT    = 10           # sentences per category, per annotator
SHARED     = 4            # of those, how many are shared (overlap for agreement)
UNIQUE     = PER_CAT - SHARED
SEED       = 42
ANNOTATORS = [
    {"id": "renana", "name": "רננה"},
    {"id": "carmit", "name": "כרמית"},
]

# fixed, logical order for the correction dropdown: acceptance -> rejection -> not-relevant
ACCEPT = [
    "קבלת עדות המתלוננת במלואה לאור רושם חיובי, כנות ומהימנות גבוהה",
    "קבלת עדות המתלוננת לאור תיאור מפורט, אותנטי וברור של מסכת האירועים",
    "קבלת גרסת המתלוננת בהיותה עדות מהימנה הנתמכת בראיות סיוע וחיזוקים",
    "קבלת עדות הקטינה כעדות אמת מהימנה המגובה בהתרשמות חוקרת הילדים",
    "העדפת עדותה המהימנה של המתלוננת ודחיית גרסת הנאשם עקב חוסר אמינותו",
    "קביעת ממצאים עובדתיים מרשיעים על בסיס אימוץ גרסת המתלוננת ודחיית הנאשם",
]
REJECT = [
    "דחיית עדות המתלוננת עקב חוסר עקביות וסתירות מהותיות בגרסתה",
    "דחיית הרשעה על בסיס עדות יחידה של המתלוננת בהעדר ראיות סיוע",
]
NEUTRAL = ["לא רלוונטי"]
CATEGORY_GROUPS = [
    {"label": "קבלת / אימוץ עדות", "items": ACCEPT},
    {"label": "דחיית עדות",        "items": REJECT},
    {"label": "אחר",               "items": NEUTRAL},
]
CLEAN_LABELS = ACCEPT + REJECT + NEUTRAL  # fixed order


# ---------- helpers ----------
def resolve_input(arg):
    """Accept an experiment id ('ex5', 'exp5', '5') OR a direct path to a CSV.
    Returns (csv_path, experiment_label)."""
    if not arg:
        return Path("classification_results.csv"), ""
    p = Path(arg)
    if p.suffix == ".csv" or p.is_file():
        label = p.resolve().parent.name
        return p, (label if re.match(r"(?i)^exp?\d+$", label) else "")
    m = re.search(r"(\d+)", arg)
    if m:
        exp = f"exp{m.group(1)}"
        return EXPERIMENTS_DIR / exp / "classification_results.csv", exp
    return Path("classification_results.csv"), ""

def _field(p, col):
    v = p.get(col)
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()

def render_html(payload):
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__PAYLOAD__*/", json.dumps(payload, ensure_ascii=False))

def suffix(exp):
    return f"_{exp}" if exp else ""


# ---------- BUILD mode ----------
def build(arg):
    csv, experiment = resolve_input(arg)
    if not Path(csv).is_file():
        sys.exit(f"CSV not found: {csv}\nPass a full path to the CSV, or check data/experiments.")
    print(f"Using: {csv}" + (f"  (experiment: {experiment})" if experiment else ""))
    print(f"Output dir: {OUT}")

    df = pd.read_csv(csv)
    df = df.dropna(subset=["sentence_id", "origin_sentence", "new_category"])
    df = df.drop_duplicates(subset=["sentence_id"])
    df["origin_sentence"] = df["origin_sentence"].astype(str).str.strip()
    df = df[df["origin_sentence"].str.len() > 0]

    vc = df["new_category"].value_counts()
    data_labels = set(vc[vc >= 5].index.tolist())
    missing = data_labels - set(CLEAN_LABELS)
    extra   = set(CLEAN_LABELS) - data_labels
    assert not missing and not extra, f"label mismatch\nmissing from order: {missing}\nnot in data: {extra}"
    df = df[df["new_category"].isin(CLEAN_LABELS)].copy()

    print(f"Clean rows: {len(df)} | categories: {len(CLEAN_LABELS)} | experiment: {experiment or '(none)'}")
    need = SHARED + 2 * UNIQUE
    for lab in CLEAN_LABELS:
        cnt = int((df["new_category"] == lab).sum())
        assert cnt >= need, f"category has only {cnt} (<{need}): {lab}"

    df.to_csv(OUT / f"cleaned_pool{suffix(experiment)}.csv", index=False)

    assignment = {a["id"]: [] for a in ANNOTATORS}
    a1, a2 = ANNOTATORS[0]["id"], ANNOTATORS[1]["id"]

    def item(r, is_shared):
        return {
            "id": str(r["sentence_id"]),
            "sentence": r["origin_sentence"],
            "assigned": r["new_category"],
            "old_category": _field(r, "category"),
            "reasoning": _field(r, "model_reasoning"),
            "shared": is_shared,
        }

    for lab in CLEAN_LABELS:
        rows = df[df["new_category"] == lab].sample(n=need, random_state=SEED).to_dict("records")
        for r in rows[:SHARED]:
            it = item(r, True); assignment[a1].append(it); assignment[a2].append(dict(it))
        for r in rows[SHARED:SHARED + UNIQUE]:
            assignment[a1].append(item(r, False))
        for r in rows[SHARED + UNIQUE:SHARED + 2 * UNIQUE]:
            assignment[a2].append(item(r, False))

    meta = {
        "config": {"per_category": PER_CAT, "shared": SHARED, "unique": UNIQUE, "seed": SEED},
        "experiment": experiment,
        "categories": CLEAN_LABELS,
        "annotators": {a["id"]: {
            "name": a["name"], "total": len(assignment[a["id"]]),
            "items": [{"sentence_id": it["id"], "assigned": it["assigned"], "shared": it["shared"]}
                      for it in assignment[a["id"]]],
        } for a in ANNOTATORS},
    }
    (OUT / f"assignment{suffix(experiment)}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shared_ids = {it["id"] for it in assignment[a1] if it["shared"]}
    print(f"Per annotator: {len(assignment[a1])} items | shared (overlap): {len(shared_ids)}")

    for a in ANNOTATORS:
        payload = {
            "annotator_id": a["id"], "annotator_name": a["name"], "experiment": experiment,
            "categories": CLEAN_LABELS, "category_groups": CATEGORY_GROUPS,
            "items": assignment[a["id"]],
        }
        out = OUT / f"annotation_{a['id']}{suffix(experiment)}.html"
        out.write_text(render_html(payload), encoding="utf-8")
        print("wrote", out.name)


# ---------- RESUME mode ----------
def resume(results_arg, csv_arg=None):
    if not results_arg:
        sys.exit("usage: python build_annotation.py resume results_*.json [classification_results.csv]")
    res = json.loads(Path(results_arg).read_text(encoding="utf-8"))
    experiment = res.get("experiment", "")

    if csv_arg:
        csv = Path(csv_arg)
    elif experiment and REPO_ROOT:
        csv = EXPERIMENTS_DIR / experiment / "classification_results.csv"
    else:
        csv = None
    if not csv or not Path(csv).is_file():
        sys.exit("CSV not found. Pass the experiment CSV as the 2nd argument.")

    df = pd.read_csv(csv).dropna(subset=["sentence_id"]).drop_duplicates("sentence_id")
    df["sentence_id"] = df["sentence_id"].astype(str)
    pool = df.set_index("sentence_id").to_dict("index")

    items = []
    for a in res["answers"]:
        sid = str(a["sentence_id"]); p = pool.get(sid, {})
        items.append({
            "id": sid, "assigned": a["assigned"],
            "old_category": _field(p, "category"),
            "reasoning": _field(p, "model_reasoning"),
            "shared": a.get("shared", False),
            "sentence": _field(p, "origin_sentence"),
        })

    unknown = {a["assigned"] for a in res["answers"]} - set(CLEAN_LABELS)
    if unknown:
        print("warning: assigned labels not in category groups:", unknown)

    answered = sum(1 for a in res["answers"] if a.get("decision") is not None)
    payload = {
        "annotator_id": res["annotator_id"],
        "annotator_name": res.get("annotator_name", res["annotator_id"]),
        "experiment": experiment,
        "categories": CLEAN_LABELS, "category_groups": CATEGORY_GROUPS,
        "items": items,
        "prefill": res["answers"],
        "only_unanswered_default": True,
    }
    out = OUT / f"annotation_{res['annotator_id']}{suffix(experiment)}_resume.html"
    out.write_text(render_html(payload), encoding="utf-8")
    print(f"wrote {out.name}")
    print(f"  {answered}/{len(items)} already answered — opens on the remaining {len(items) - answered}.")


# ---------- COMPARE mode ----------
def _pool_from_csv(csv):
    df = pd.read_csv(csv).drop_duplicates("sentence_id")
    df["sentence_id"] = df["sentence_id"].astype(str)
    return df.set_index("sentence_id").to_dict("index")

def compare(json_a, json_b, csv_arg=None):
    if not json_a or not json_b:
        sys.exit("usage: python build_annotation.py compare results_A.json results_B.json [classification_results.csv]")
    a = json.loads(Path(json_a).read_text(encoding="utf-8"))
    b = json.loads(Path(json_b).read_text(encoding="utf-8"))
    experiment = a.get("experiment", "") or b.get("experiment", "")

    if csv_arg:
        csv = Path(csv_arg)
    elif experiment and REPO_ROOT:
        csv = EXPERIMENTS_DIR / experiment / "classification_results.csv"
    else:
        csv = None
    if not csv or not Path(csv).is_file():
        sys.exit("CSV not found. Pass the experiment CSV as the 3rd argument.")
    pool = _pool_from_csv(csv)

    da = {x["sentence_id"]: x for x in a["answers"] if x.get("decision")}
    db = {x["sentence_id"]: x for x in b["answers"] if x.get("decision")}
    shared = [s for s in da if s in db]   # answered by both

    def final(x):
        v = x["corrected"] if x["decision"] == "incorrect" else x["assigned"]
        return v if v else "(לא נבחרה)"

    items = []
    for sid in shared:
        xa, xb = da[sid], db[sid]
        p = pool.get(str(sid), {})
        differ = (xa["decision"] != xb["decision"]) or (final(xa) != final(xb))
        items.append({
            "id": sid,
            "sentence": _field(p, "origin_sentence"),
            "assigned": xa["assigned"],
            "old_category": _field(p, "category"),
            "reasoning": _field(p, "model_reasoning"),
            "r": {"decision": xa["decision"], "final": final(xa), "note": xa.get("note", "")},
            "c": {"decision": xb["decision"], "final": final(xb), "note": xb.get("note", "")},
            "differ": differ,
        })
    n_diff = sum(1 for it in items if it["differ"])
    payload = {
        "experiment": experiment,
        "annotators": [
            {"id": a["annotator_id"], "name": a.get("annotator_name", a["annotator_id"])},
            {"id": b["annotator_id"], "name": b.get("annotator_name", b["annotator_id"])},
        ],
        "items": items,
    }
    tpl = (SCRIPT_DIR / "compare_template.html").read_text(encoding="utf-8")
    out = OUT / f"compare{suffix(experiment)}.html"
    out.write_text(tpl.replace("/*__PAYLOAD__*/", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"wrote {out.name}")
    print(f"  {len(items)} shared sentences | {n_diff} where they differ (opens on these).")


# ---------- dispatch ----------
if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "resume":
        resume(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None)
    elif args and args[0] == "compare":
        compare(args[1] if len(args) > 1 else None,
                args[2] if len(args) > 2 else None,
                args[3] if len(args) > 3 else None)
    else:
        build(args[0] if args else None)
