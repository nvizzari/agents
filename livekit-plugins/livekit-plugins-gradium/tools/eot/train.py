"""Train, evaluate and export the learned end-of-turn scorer.

    python tools/eot/train.py report    # every comparison in the plan's results section
    python tools/eot/train.py policy    # passive-only weights for `collect.py --mode flush`
    python tools/eot/train.py export    # bundled weights + the test fixture
    python tools/eot/train.py export-held-out --out DIR   # one file per language, for eot-bench

Features come from running `FeatureWindow` (the plugin's own code) over the recorded
streams, so training and live scoring share one implementation. A score is read at each
100 ms grid point of each annotated silence span from the last step with
`audio_time <= timestamp`, then scored with the eot-bench policy sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (  # noqa: E402
    DATA_DIR,
    DATASET,
    FIXTURE_PATH,
    FLUSH_SENT,
    LANGUAGES,
    REVISION,
    WEIGHTS_DIR,
    Driver,
    default_bench_path,
    import_harness,
    iter_rows,
    raw_path,
    read_records,
)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from livekit.plugins.gradium.experimental.scorer import (  # noqa: E402
    FEATURES,
    FORMAT_VERSION,
    FeatureWindow,
)
from livekit.plugins.gradium.experimental.signals import Step, Word  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
import_harness()
from eot_harness.io import DEFAULT_MIN_SILENCE, _time_grid, span_label  # noqa: E402
from eot_harness.metrics import (  # noqa: E402
    compute_metrics_from_predictions,
    get_pareto_operating_points,
)

EPS = 1e-6
NON_EN = tuple(lang for lang in LANGUAGES if lang != "en")
COHORTS = ("general", "accented", "short_turns")
PASSIVE_FEATURES = [f for f in FEATURES if f != "steps_since_flush"]
POST_FLUSH_FEATURES = list(FEATURES)
PARAMS: dict[str, Any] = {
    "max_iter": 100,
    "learning_rate": 0.1,
    "max_leaf_nodes": 24,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}
META = ["id", "language", "cohort", "span_index", "timestamp", "silence_dur", "label"]
BENCH_OUT = "output/livekit__eot-bench-data__validation__min_silence_100ms"
REFERENCES = {
    "LiveKit v1.1 (committed)": "live_kit_turn_detector_adapter__turn_detector_v1.1",
    "Gradium (committed)": "gradium_streaming_adapter__default",
}


# --- feature table ------------------------------------------------------------------


class _Recorder:
    """Scorer-shaped wrapper that keeps the feature vector after every step."""

    def __init__(self) -> None:
        self.window = FeatureWindow()
        self.times: list[float] = []
        self.vectors: list[list[float]] = []
        self.flushed: list[bool] = []

    def on_step(self, step: Step) -> float:
        self.vectors.append(self.window.on_step(step))
        self.times.append(step.audio_time)
        self.flushed.append(self.window.flushed)
        return 0.0

    def on_word(self, word: Word, audio_time: float) -> None:
        self.window.on_word(word, audio_time)

    def on_flush(self) -> None:
        self.window.on_flush()


def spans_by_id(langs: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    return {
        r["id"]: r["silence_spans"]
        for lang in langs
        for r in iter_rows(lang, ["id", "silence_spans"])
    }


def build_table(mode: str, langs: Iterable[str]) -> pd.DataFrame:
    """One row per (turn, span, grid point) with the features of the step scored there."""
    langs = list(langs)
    spans_of = spans_by_id(langs)
    meta: list[tuple[Any, ...]] = []
    feats: list[list[float]] = []
    flags: list[tuple[bool, bool]] = []
    zero = [0.0] * len(FEATURES)
    for rec in read_records(mode, langs):
        rec_ = _Recorder()
        driver = Driver(rec_)
        for msg in rec["messages"]:
            driver.feed(msg)
        times = np.asarray(rec_.times)
        spans = spans_of[rec["id"]]
        for si, span in enumerate(spans):
            start, end = float(span["start"]), float(span["end"])
            if end - start < DEFAULT_MIN_SILENCE - EPS:
                continue
            label = span_label(spans, si)
            for t in _time_grid(start, end, 0.1):
                j = int(np.searchsorted(times, t + EPS, side="right")) - 1
                meta.append(
                    (rec["id"], rec["language"], rec["cohort"], si, t, round(t - start, 6), label)
                )
                feats.append(rec_.vectors[j] if j >= 0 else zero)
                flags.append((j >= 0, rec_.flushed[j] if j >= 0 else False))
    df = pd.DataFrame(meta, columns=META)
    df[list(FEATURES)] = np.asarray(feats, dtype=np.float64)
    df["has_step"] = [f[0] for f in flags]
    df["flushed"] = [f[1] for f in flags]
    df["y"] = (df["label"] == "eot").astype(int)
    return df


# --- models -------------------------------------------------------------------------


def fit(df: pd.DataFrame, features: list[str], **params: Any) -> HistGradientBoostingClassifier:
    rows = df[df["has_step"]]
    model = HistGradientBoostingClassifier(**{**PARAMS, **params})
    model.fit(rows[features].to_numpy(), rows["y"].to_numpy())
    return model


def predict(
    model: HistGradientBoostingClassifier, df: pd.DataFrame, features: list[str]
) -> np.ndarray:
    p = model.predict_proba(df[features].to_numpy())[:, 1]
    return np.where(df["has_step"].to_numpy(), p, 0.0)


def oof(df: pd.DataFrame, features: list[str], folds: int = 5) -> np.ndarray:
    """Out-of-fold scores, folds grouped by turn: no turn is scored by a model that saw it."""
    out = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=folds).split(df, groups=df["id"]):
        out[te] = predict(fit(df.iloc[tr], features), df.iloc[te], features)
    return out


def fallback(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["has_step"], np.minimum(df["p2"], df["p3"]), 0.0)


def export_forest(model: HistGradientBoostingClassifier, features: list[str]) -> dict[str, Any]:
    """scikit-learn trees to the weights-file layout read by `scorer._Forest`."""
    trees = []
    for (predictor,) in model._predictors:
        nodes = predictor.nodes
        leaf = nodes["is_leaf"].astype(bool)
        trees.append(
            {
                "feature": np.where(leaf, -1, nodes["feature_idx"].astype(np.int64))
                .astype(int)
                .tolist(),
                "threshold": np.where(leaf, 0.0, nodes["num_threshold"]).astype(float).tolist(),
                "left": np.where(leaf, -1, nodes["left"].astype(np.int64)).astype(int).tolist(),
                "right": np.where(leaf, -1, nodes["right"].astype(np.int64)).astype(int).tolist(),
                "value": np.where(leaf, nodes["value"], 0.0).astype(float).tolist(),
            }
        )
    return {
        "features": features,
        "baseline": float(np.ravel(model._baseline_prediction)[0]),
        "trees": trees,
    }


# --- benchmark ----------------------------------------------------------------------


def on_plan_grid(df: pd.DataFrame) -> np.ndarray:
    """Grid points on the 100 ms grid, without the span end the harness appends.

    The plan's evidence table was computed on this grid. The harness grid (and every
    committed prediction file) also scores the span end itself, which gives the score one
    more chance to cut off a hold: about 80 ms more latency at the 5% budget for Gradium.
    """
    sd = df["silence_dur"].to_numpy()
    return np.isclose(np.round(sd * 10) / 10, sd, atol=1e-4)


def bench(
    df: pd.DataFrame, score: np.ndarray | pd.Series, *, plan_grid: bool = False
) -> dict[str, float]:
    preds = df[["id", "span_index", "timestamp", "silence_dur", "label"]].copy()
    preds["p_eot"] = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    if plan_grid:
        preds = preds[on_plan_grid(preds)]
    return bench_predictions(preds)


def bench_predictions(preds: pd.DataFrame) -> dict[str, float]:
    trade, _ = compute_metrics_from_predictions(preds)
    lat = get_pareto_operating_points(trade, "cutoff_rate", [0.02, 0.05, 0.10])
    cut = get_pareto_operating_points(trade, "mean_latency", [0.3, 0.6])
    return {
        "lat2": lat.value.iloc[0] * 1000,
        "lat5": lat.value.iloc[1] * 1000,
        "lat10": lat.value.iloc[2] * 1000,
        "cut300": cut.value.iloc[0] * 100,
        "cut600": cut.value.iloc[1] * 100,
    }


def reference(bench_root: Path, lang: str, run: str, ids: set[str]) -> dict[str, float] | None:
    path = bench_root / BENCH_OUT / lang / run / "predictions.parquet"
    if not path.exists():
        return None
    p = pd.read_parquet(path)
    return bench_predictions(p[p["id"].isin(ids)])


def fmt_row(name: str, m: dict[str, float] | None) -> str:
    if m is None:
        return f"| {name} | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {name} | {m['lat2']:.0f} ms | {m['lat5']:.0f} ms | {m['lat10']:.0f} ms "
        f"| {m['cut300']:.1f}% | {m['cut600']:.1f}% |"
    )


TABLE_HEAD = (
    "| Score | Latency @ 2% | Latency @ 5% | Latency @ 10% | Cut-offs @ 300 ms "
    "| Cut-offs @ 600 ms |\n|---|---|---|---|---|---|"
)


def table(title: str, rows: list[tuple[str, dict[str, float] | None]]) -> str:
    return "\n".join([f"\n{title}\n", TABLE_HEAD, *(fmt_row(n, m) for n, m in rows)])


# --- tables -------------------------------------------------------------------------


def tables(mode: str, rebuild: bool) -> pd.DataFrame:
    """Feature table of every recorded language, cached until a raw file changes.

    Raises:
        FileNotFoundError: A language has not been collected in this mode.
    """
    sources = [raw_path(mode, lang) for lang in LANGUAGES]
    cache = DATA_DIR / "derived" / f"{mode}.parquet"
    fresh = cache.exists() and all(
        s.stat().st_mtime < cache.stat().st_mtime for s in sources if s.exists()
    )
    if fresh and not rebuild:
        return pd.read_parquet(cache)
    df = build_table(mode, LANGUAGES)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# --- commands -----------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    df = tables("passive", args.rebuild)
    en, non_en = df[df.language == "en"], df[df.language != "en"]
    ids = set(en["id"])
    print(
        f"passive: {df['id'].nunique()} turns, {len(df)} grid rows; English "
        f"{len(ids)} turns, spans {en.drop_duplicates(['id', 'span_index']).label.value_counts().to_dict()}"
    )

    scores: dict[str, np.ndarray] = {}
    scores["Fallback `min(p2, p3)`"] = fallback(en)
    xling = fit(non_en, PASSIVE_FEATURES)
    scores["Learned, trained on 13 other languages (held out)"] = predict(
        xling, en, PASSIVE_FEATURES
    )
    scores["Learned, 5-fold OOF on English"] = oof(en, PASSIVE_FEATURES)
    all_oof = oof(df, PASSIVE_FEATURES)
    scores["Learned, 5-fold OOF on all 14 languages"] = all_oof[(df.language == "en").to_numpy()]

    results: dict[str, Any] = {"english": {}, "cohorts": {}, "languages": {}, "ablations": {}}
    refs = {n: reference(args.bench, "en", run, ids) for n, run in REFERENCES.items()}
    rows = [(n, bench(en, s)) for n, s in scores.items()]
    results["english"] = dict(rows) | refs
    print(table("English, all cohorts, harness grid", rows + list(refs.items())))
    prows = [(n, bench(en, s, plan_grid=True)) for n, s in scores.items()]
    results["english_plan_grid"] = dict(prows)
    print(table("English, all cohorts, plan grid (no span-end point)", prows))

    for cohort in COHORTS:
        mask = (en.cohort == cohort).to_numpy()
        sub = en[mask]
        crow = [(n, bench(sub, s[mask])) for n, s in scores.items()]
        cref = [
            (n, reference(args.bench, "en", run, set(sub["id"]))) for n, run in REFERENCES.items()
        ]
        results["cohorts"][cohort] = dict(crow + cref)
        print(table(f"English, cohort `{cohort}` ({sub['id'].nunique()} turns)", crow + cref))

    # per language: leave-one-language-out keeps every number held out
    lrows = []
    for lang in LANGUAGES:
        held = df[df.language == lang]
        model = xling if lang == "en" else fit(df[df.language != lang], PASSIVE_FEATURES)
        lids = set(held["id"])
        m = {
            "fallback": bench(held, fallback(held)),
            "learned": bench(held, predict(model, held, PASSIVE_FEATURES)),
            **{n: reference(args.bench, lang, run, lids) for n, run in REFERENCES.items()},
        }
        results["languages"][lang] = m
        lrows.append((lang, m))
        print(lang, {k: (round(v["lat5"]) if v else None) for k, v in m.items()}, flush=True)
    print(
        "\n| Language | Fallback @5% / @10% | Learned (other 13) @5% / @10% | Gradium committed @5% / @10% | v1.1 committed @5% / @10% |"
    )
    print("|---|---|---|---|---|")
    for lang, m in lrows:
        cells = [
            "n/a" if v is None else f"{v['lat5']:.0f} / {v['lat10']:.0f} ms"
            for v in (
                m["fallback"],
                m["learned"],
                m["Gradium (committed)"],
                m["LiveKit v1.1 (committed)"],
            )
        ]
        print(f"| {lang} | " + " | ".join(cells) + " |")

    # feature ablations and tree budget, cross-language
    ablations = {
        "all passive features": PASSIVE_FEATURES,
        "without word timing": [f for f in PASSIVE_FEATURES if f.startswith("p")],
        "4 horizons only": ["p0", "p1", "p2", "p3"],
    }
    arows = []
    for name, feats in ablations.items():
        arows.append((name, bench(en, predict(fit(non_en, feats), en, feats))))
    big = fit(non_en, PASSIVE_FEATURES, max_iter=300, learning_rate=0.06)
    arows.append(("300 trees, learning rate 0.06", bench(en, predict(big, en, PASSIVE_FEATURES))))
    results["ablations"] = dict(arows)
    print(table("Cross-language ablations (English, held out)", arows))

    out = DATA_DIR / "results_passive.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


def flush_points(mode: str, langs: Iterable[str]) -> pd.DataFrame:
    """Where each flush landed: span, label, silence at the flush.

    Labels: `hold` / `eot` inside an annotated span, `leading` before the first word of
    the stream arrived, `speech` anywhere else (speech, or a pause shorter than a span).
    """
    langs = list(langs)
    spans_of = spans_by_id(langs)
    rows = []
    for rec in read_records(mode, langs):
        driver = Driver(_Recorder())
        heard = False
        for msg in rec["messages"]:
            heard = heard or msg.get("type") == "text"
            if msg.get("type") == FLUSH_SENT:
                t = driver.audio_time
                spans = spans_of[rec["id"]]
                hit = [
                    (si, s) for si, s in enumerate(spans) if s["start"] - EPS <= t <= s["end"] + EPS
                ]
                si, span = hit[0] if hit else (None, None)
                rows.append(
                    {
                        "id": rec["id"],
                        "language": rec["language"],
                        "t": t,
                        "span_index": si,
                        "label": span_label(spans, si)
                        if span
                        else ("speech" if heard else "leading"),
                        "silence": t - span["start"] if span else None,
                    }
                )
            driver.feed(msg)
    return pd.DataFrame(rows)


def cmd_report_flush(args: argparse.Namespace) -> None:
    passive = tables("passive", args.rebuild)
    flush = tables("flush", args.rebuild)
    key = ["id", "span_index", "timestamp"]
    p_en = passive[passive.language == "en"].reset_index(drop=True)
    f_en = flush[flush.language == "en"].reset_index(drop=True)

    pas = fit(passive[passive.language != "en"], PASSIVE_FEATURES)
    post = fit(flush[(flush.language != "en") & flush.flushed], POST_FLUSH_FEATURES)
    pas_score = predict(pas, p_en, PASSIVE_FEATURES)
    hybrid = np.where(
        f_en["flushed"],
        predict(post, f_en, POST_FLUSH_FEATURES),
        predict(pas, f_en, PASSIVE_FEATURES),
    )
    contaminated = predict(pas, f_en, PASSIVE_FEATURES)

    pts = flush_points("flush", ["en"])
    spans = p_en.drop_duplicates(["id", "span_index"])
    n_spans = spans.label.value_counts()
    flushed_spans = pts.dropna(subset=["span_index"]).drop_duplicates(["id", "span_index"])
    print(
        f"English flushes: {len(pts)} in {pts['id'].nunique()} turns; by span label {pts.label.value_counts().to_dict()}"
    )
    for lbl in ("hold", "eot"):
        got = flushed_spans[flushed_spans.label == lbl]
        print(
            f"  {lbl}: {len(got)}/{n_spans[lbl]} spans flushed ({100 * len(got) / n_spans[lbl]:.1f}%), "
            f"silence at flush p50 {got.silence.median():.2f} s, p90 {got.silence.quantile(0.9):.2f} s"
        )
    print(
        f"  turns with a wasted flush (in a hold): {pts[pts.label == 'hold']['id'].nunique()}/{p_en['id'].nunique()}"
    )

    # a flush must leave no trace once the speaker resumed: compare unflushed rows
    same = f_en[~f_en.flushed][key + ["p2"]].merge(p_en[key + ["p2"]], on=key)
    print(
        f"  unflushed grid points: {len(same)}, mean |p2 flush - p2 passive| "
        f"{np.abs(same.p2_x - same.p2_y).mean():.4f}; flushed grid points: "
        f"{int(f_en.flushed.sum())}"
    )

    # separation on the same grid points: post-flush steps vs the passive stream
    m = f_en[f_en.flushed][key + ["y"]].assign(
        post=hybrid[f_en.flushed.to_numpy()], contam=contaminated[f_en.flushed.to_numpy()]
    )
    m = m.merge(p_en[key].assign(passive=pas_score), on=key)
    print(f"\npost-flush grid points: {len(m)} ({m.y.mean() * 100:.0f}% in end-of-turn spans)")
    print(f"  ROC AUC, passive score on the passive stream: {roc_auc_score(m.y, m.passive):.3f}")
    print(f"  ROC AUC, post_flush score on the flush stream: {roc_auc_score(m.y, m.post):.3f}")
    print(f"  ROC AUC, passive weights on the flush stream:  {roc_auc_score(m.y, m.contam):.3f}")

    results = {}
    for plan_grid in (False, True):
        rows = [
            ("Passive only (passive stream)", bench(p_en, pas_score, plan_grid=plan_grid)),
            ("Passive + post_flush (flush stream)", bench(f_en, hybrid, plan_grid=plan_grid)),
            (
                "Passive weights on the flush stream (no post_flush set)",
                bench(f_en, contaminated, plan_grid=plan_grid),
            ),
        ]
        grid = "plan grid" if plan_grid else "harness grid"
        results[grid] = dict(rows)
        print(table(f"English, held out, {grid}: does the flush hurt the decision?", rows))
    (DATA_DIR / "results_flush.json").write_text(json.dumps(results, indent=1))


def training_meta(langs: list[str], modes: list[str], df: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "split": "validation",
        "languages": langs,
        "streams": modes,
        "turns": int(df["id"].nunique()),
        "grid_rows": int(len(df)),
    }


def write_weights(
    name: str, description: str, langs: list[str], args: argparse.Namespace, *, with_flush: bool
) -> dict[str, Any]:
    passive = tables("passive", args.rebuild)
    p_rows = passive[passive.language.isin(langs)]
    pas = fit(p_rows, PASSIVE_FEATURES)
    models = {"passive": (pas, PASSIVE_FEATURES)}
    training = {"passive": training_meta(langs, ["passive"], p_rows)}
    if with_flush:
        flush = tables("flush", args.rebuild)
        f_rows = flush[flush.language.isin(langs) & flush.flushed]
        models["post_flush"] = (fit(f_rows, POST_FLUSH_FEATURES), POST_FLUSH_FEATURES)
        training["post_flush"] = training_meta(langs, ["flush"], f_rows)
    else:
        models["post_flush"] = models["passive"]
        training["post_flush"] = {"note": "copy of passive: no flush data yet"}
    ready = recorded_server(langs, ["passive", "flush"] if with_flush else ["passive"])
    spec = {
        "format": FORMAT_VERSION,
        "name": name,
        "description": description,
        "model": ready["model_ext"],
        "delay_in_frames": ready["delay_in_frames"],
        "training": training,
        "params": {k: v for k, v in PARAMS.items() if k != "early_stopping"},
        "sets": {k: export_forest(m, f) for k, (m, f) in models.items()},
    }
    return {"spec": spec, "models": models}


def recorded_server(langs: list[str], modes: list[str]) -> dict[str, Any]:
    """The `ready` message every training stream shares.

    Weights are tied to one server model and model delay. A collection that straddles a
    server update would silently mix two models, so it is refused.

    Raises:
        SystemExit: The recordings come from more than one model or model delay.
    """
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for mode in modes:
        for rec in read_records(mode, langs):
            ready = rec["ready"]
            seen.setdefault((ready["model_ext"], int(ready["delay_in_frames"])), ready)
    if len(seen) != 1:
        raise SystemExit(
            f"recordings span several (model_ext, delay_in_frames): {sorted(seen)}; "
            "re-collect them against one server model"
        )
    return next(iter(seen.values()))


def save(spec: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")


def cmd_policy(args: argparse.Namespace) -> None:
    out = write_weights(
        "policy-passive-xling",
        "passive weights for the flush collector",
        list(NON_EN),
        args,
        with_flush=False,
    )
    save(out["spec"], DATA_DIR / "policy.json")


def write_fixture(models: dict[str, Any], weights_name: str) -> None:
    """Pin a few held-out English turns and the scikit-learn score at every step."""
    picks = []
    for mode, want in (("passive", 3), ("flush", 2)):
        for rec in read_records(mode, ["en"]):
            has_flush = any(m.get("type") == FLUSH_SENT for m in rec["messages"])
            if rec["audio_s"] < 6.0 and (mode == "passive" or has_flush):
                picks.append(rec)
                want -= 1
            if want == 0:
                break
    turns = []
    for rec in picks:
        r = _Recorder()
        driver = Driver(r)
        events: list[list[Any]] = []
        for msg in rec["messages"]:
            step = driver.feed(msg)
            if msg.get("type") == FLUSH_SENT:
                events.append(["flush"])
            elif step is not None:
                events.append(["step", step.audio_time, *step.probs])
            elif msg.get("type") == "text":
                events.append(["word", msg["text"], msg["start_s"]])
        x = np.asarray(r.vectors)
        flushed = np.asarray(r.flushed)
        expected = np.empty(len(x))
        for key, mask in (("passive", ~flushed), ("post_flush", flushed)):
            if mask.any():
                model, feats = models[key]
                cols = [FEATURES.index(f) for f in feats]
                expected[mask] = model.predict_proba(x[mask][:, cols])[:, 1]
        turns.append(
            {"id": rec["id"], "mode": rec["mode"], "events": events, "expected": expected.tolist()}
        )
    FIXTURE_PATH.write_text(
        json.dumps({"weights": weights_name, "turns": turns}, separators=(",", ":")) + "\n"
    )
    print(f"wrote {FIXTURE_PATH} ({sum(len(t['expected']) for t in turns)} steps)")


def cmd_export(args: argparse.Namespace) -> None:
    default = write_weights(
        "gradium-eot-xling-v1",
        "Learned end-of-turn score trained on the 13 non-English languages of eot-bench "
        "validation; English is held out.",
        list(NON_EN),
        args,
        with_flush=True,
    )
    save(default["spec"], WEIGHTS_DIR / "default.json")
    full = write_weights(
        "gradium-eot-all-v1",
        "Learned end-of-turn score trained on all 14 languages of eot-bench validation, "
        "English included: its English benchmark numbers are not held out.",
        list(LANGUAGES),
        args,
        with_flush=True,
    )
    save(full["spec"], WEIGHTS_DIR / "all-languages.json")
    write_fixture(default["models"], "default")


def cmd_export_held_out(args: argparse.Namespace) -> None:
    """Leave-one-language-out weights: `<lang>.json` trained on the 13 other languages.

    The eot-bench adapter scores each language with the file that never saw it (English
    uses the bundled `default`, which is the same thing). Never ship these in the plugin.
    """
    out_dir = Path(args.out)
    for lang in args.lang or [lang for lang in LANGUAGES if lang != "en"]:
        others = [x for x in LANGUAGES if x != lang]
        spec = write_weights(
            f"gradium-eot-loo-{lang}-v1",
            f"Learned end-of-turn score trained on the 13 eot-bench validation languages "
            f"other than {lang!r}; {lang!r} is held out. Same features and parameters as the "
            "bundled default weights.",
            others,
            args,
            with_flush=True,
        )["spec"]
        save(spec, out_dir / f"{lang}.json")


def cmd_timing(args: argparse.Namespace) -> None:
    """Wall time of `LearnedScorer.on_step` (features + trees) on recorded English streams."""
    import time

    from livekit.plugins.gradium.experimental.scorer import LearnedScorer, load_weights

    weights = load_weights("default")
    per_step: list[float] = []
    for mode in ("passive", "flush"):
        for rec in read_records(mode, ["en"]):
            scorer = LearnedScorer(weights)
            driver = Driver(scorer)
            for msg in rec["messages"]:
                if msg.get("type") != "step":
                    driver.feed(msg)
                    continue
                t0 = time.perf_counter()
                driver.feed(msg)
                per_step.append(time.perf_counter() - t0)
    us = np.asarray(per_step) * 1e6
    print(
        f"{len(us)} steps (parse + features + trees): mean {us.mean():.0f} us, "
        f"p50 {np.median(us):.0f} us, p99 {np.quantile(us, 0.99):.0f} us, max {us.max():.0f} us"
    )


def cmd_policy_sim(args: argparse.Namespace) -> None:
    """How often EagerPolicy would flush on the passive streams, per span label."""
    from collect import EagerPolicy, EagerSettings

    from livekit.plugins.gradium.experimental.scorer import LearnedScorer, load_weights

    weights = load_weights(DATA_DIR / "policy.json")
    langs = args.lang or ["en"]
    spans_of = spans_by_id(langs)
    for eager in args.eager:
        settings = EagerSettings(args.pause_threshold, eager, args.min_silence, 0.16)
        hits = {"hold": 0, "eot": 0}
        totals = {"hold": 0, "eot": 0}
        silences = {"hold": [], "eot": []}
        for rec in read_records("passive", langs):
            scorer = LearnedScorer(weights)
            policy = EagerPolicy(scorer, settings)
            driver = Driver(scorer)
            spans = spans_of[rec["id"]]
            fired: dict[int, float] = {}
            for msg in rec["messages"]:
                step = driver.feed(msg)
                policy.on_message(msg)
                if step is not None and policy.should_flush(step, driver.score):
                    t = step.audio_time
                    for si, s in enumerate(spans):
                        if s["start"] - EPS <= t <= s["end"] + EPS and si not in fired:
                            fired[si] = t - s["start"]
                    # the passive stream never resumes the scorer; emulate one flush per pause
                    driver.feed({"type": FLUSH_SENT})
            for si, s in enumerate(spans):
                if s["end"] - s["start"] < DEFAULT_MIN_SILENCE - EPS:
                    continue
                lbl = span_label(spans, si)
                totals[lbl] += 1
                if si in fired:
                    hits[lbl] += 1
                    silences[lbl].append(fired[si])
        print(
            f"eager={eager:.2f}: holds flushed {100 * hits['hold'] / totals['hold']:.1f}%, "
            f"ends flushed {100 * hits['eot'] / totals['eot']:.1f}% "
            f"(silence at flush p50 {np.median(silences['eot']):.2f} s)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="recompute the cached feature tables")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    sub.add_parser("report-flush")
    sub.add_parser("policy")
    sub.add_parser("export")
    held = sub.add_parser("export-held-out")
    held.add_argument("--out", required=True, help="directory for <lang>.json")
    held.add_argument("--lang", nargs="+", choices=LANGUAGES, help="default: all but en")
    sub.add_parser("timing")
    sim = sub.add_parser("policy-sim")
    sim.add_argument("--lang", nargs="+", choices=LANGUAGES)
    sim.add_argument("--eager", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6])
    sim.add_argument("--pause-threshold", type=float, default=0.2)
    sim.add_argument("--min-silence", type=float, default=0.3)
    args = ap.parse_args()
    args.bench = default_bench_path()
    {
        "report": cmd_report,
        "report-flush": cmd_report_flush,
        "policy": cmd_policy,
        "export": cmd_export,
        "export-held-out": cmd_export_held_out,
        "policy-sim": cmd_policy_sim,
        "timing": cmd_timing,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
