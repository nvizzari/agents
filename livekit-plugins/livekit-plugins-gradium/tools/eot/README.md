# Gradium end-of-turn tools: data, scorer, simulator, presets

Scripts that produce the scorer weights in
`livekit/plugins/gradium/experimental/weights/`, the test fixture
`tests/gradium_scorer_steps.json`, and the preset values in
`experimental/profiles.py`. They are not part of the wheel. Results and the reasoning
behind the choices are in `../../STT_EXPERIMENTAL_PLAN.md`, sections "Results: scorer
(steps 2-3)" and "Results: simulator and presets (step 4)".

Everything under `data/` is generated, git-ignored, and can be deleted.

| Script | What it does | Needs the API |
|---|---|---|
| `download.py` | eot-bench validation parquet files at the pinned revision | no (Hugging Face) |
| `collect.py` | replays turns through production Gradium and records every message | yes |
| `train.py` | trains, evaluates and exports the scorer weights | no |
| `simulate.py` | replays recordings through the plugin's `TurnMachine`; calibrates presets | no |
| `live_check.py` | end-to-end check of `STTExperimental` at real-time pacing | yes |
| `noise.py` | pink and babble noise for the noisy replays (a library) | no |

## Environment

Run from the repository root, in the workspace environment plus the research
dependencies, so `livekit.plugins.gradium.experimental` is the code in this checkout:

```bash
alias eot='uv run --no-sync --with scikit-learn --with pandas --with pyarrow \
  --with librosa --with soundfile --with huggingface_hub python -I'
```

Scoring uses an [eot-bench](https://github.com/livekit/eot-bench) checkout, found at
`$EOT_BENCH` or `~/git/oss/eot-bench`, for its span grid, policy sweep and committed
reference predictions. Audio is resampled with the bench's own code, so the turns sent to
Gradium are byte-identical to an eot-bench run.

`collect.py` and `live_check.py` need `GRADIUM_API_KEY` in the environment or in
`~/.bench.env`. The key is never printed.

## Re-computing the scorer weights (runbook)

Follow this when Gradium ships a new server model, when the server starts accepting a
language the weights were not trained with, or when the model delay changes. It needs no
knowledge of the code. Every command runs from the repository root with the `eot` alias.

### 1. Is it needed?

The weights are tied to one server model and one model delay. Re-compute when any of
these holds:

- **The server model changed.** Each session starts with a `ready` message whose
  `model_ext` names the model. The weights record the model they were trained against:

  ```bash
  eot -c "from livekit.plugins.gradium.experimental.scorer import load_weights; \
  w = load_weights('default'); print(w.model, w.meta['delay_in_frames'])"
  # 933b5345@150 10   (current weights)
  ```

  When the two differ, the plugin keeps running on the `min(p2, p3)` fallback with the
  fallback's own preset thresholds, and logs once per stream, at warning level:

  ```text
  Gradium scorer weights 'gradium-eot-xling-v1' were trained for model '933b5345@150' but the server runs '<new model>', using the min(p2, p3) fallback
  ```

  Nothing breaks, but turn-taking is slower: on English, 1,060 ms instead of 874 ms mean
  latency at a 5% false cut-off budget on the harness grid.
- **A new language.** The server accepts a hint for a benchmark language that was
  collected as `any` (`ANY_HINT_LANGUAGES` in `collect.py`). Its recordings change, so its training data must
  be collected again.
- **The model delay changed.** The presets send `delay_in_frames=10`, and the word-timing
  features were learned at that delay. Changing the preset's delay, or a change of what
  the server does with it, needs new data at the new delay.

### 2. Prerequisites

- **API key**: production `GRADIUM_API_KEY`. One key allows 12 concurrent sessions; a
  13th is refused with close code 1008. `collect.py` retries refused sessions with a
  back-off. Leave room if anything else uses the key: `--concurrency 6` is the default.
- **Dataset**: `livekit/eot-bench-data`, validation split, pinned at revision
  `ae834af9e9bddf73d1f550c190aaeb99bf31f523` (`common.REVISION`). The `full` revision is
  not public; the validation split is the only training data. About 2 GB.
- **Volume**: 7,342 turns in 14 languages, 19.75 hours of audio, sent twice (passive and
  flush mode). At 12 sessions, English (2,288 turns) took 10 min
  passive and 12.6 min in flush mode, so a full pass takes about 35 min passive and
  45 min flush at 12 sessions, twice that at 6.
- **Disk**: about 2.3 GB under `data/` (parquet, recordings, cached feature tables).

### 3. Steps

1. **Download** (once per dataset revision):

   ```bash
   eot livekit-plugins/livekit-plugins-gradium/tools/eot/download.py
   ```

2. **Collect passive streams.** Audio is sent as fast as the socket takes it; the
   probabilities do not depend on pacing. Move the old recordings away first:
   `collect.py` appends and skips turns already recorded.

   ```bash
   mv livekit-plugins/livekit-plugins-gradium/tools/eot/data/raw \
      livekit-plugins/livekit-plugins-gradium/tools/eot/data/raw-<old model>
   eot livekit-plugins/livekit-plugins-gradium/tools/eot/collect.py --mode passive \
     --concurrency 12 --delay-in-frames 10
   ```

3. **Train the passive policy weights** that decide where flush mode flushes:

   ```bash
   eot livekit-plugins/livekit-plugins-gradium/tools/eot/train.py --rebuild policy
   # writes data/policy.json (passive set only, trained on the 13 non-English languages)
   ```

4. **Collect flush-mode streams.** Audio is sent in lockstep, 160 ms ahead of the last
   step received (`--lead`). The passive score from step 3 opens a pause at 0.2
   (`--pause-threshold`); the first step at least 0.3 s into it (`--min-silence`) that
   scores 0.5 or more (`--eager-threshold`) sends a `flush`, and audio stops until
   `flushed`. One flush per pause, none before the first word. These are the training
   data for the `post_flush` weights. They depend on the passive score that chose the
   flush points, so **flush mode must be collected again whenever the passive weights
   change**, including after a server model change.

   ```bash
   eot livekit-plugins/livekit-plugins-gradium/tools/eot/collect.py --mode flush \
     --weights livekit-plugins/livekit-plugins-gradium/tools/eot/data/policy.json \
     --concurrency 12 --delay-in-frames 10
   ```

5. **Evaluate** before exporting (prints the tables of the plan's results section and
   writes `data/results_passive.json`, `data/results_flush.json`):

   ```bash
   eot livekit-plugins/livekit-plugins-gradium/tools/eot/train.py --rebuild report
   eot livekit-plugins/livekit-plugins-gradium/tools/eot/train.py report-flush
   ```

6. **Export the bundled weights and the test fixture:**

   ```bash
   eot livekit-plugins/livekit-plugins-gradium/tools/eot/train.py export
   ```

   This writes:
   - `experimental/weights/default.json` (`gradium-eot-xling-v1`): trained on the 13
     non-English languages. English is never seen, so English numbers are held out.
     The plugin loads it under the name `default`.
   - `experimental/weights/all-languages.json` (`gradium-eot-all-v1`): trained on all 14
     languages. It is in-sample for every benchmark language: never publish benchmark
     numbers computed with it.
   - `tests/gradium_scorer_steps.json`: the parity fixture (step 7).

   The file layout version (`format`) is `FORMAT_VERSION` in `experimental/scorer.py`;
   bump it only when the layout changes, since the plugin rejects any other version. The
   recorded `model` and `delay_in_frames` are copied from the `ready` messages of the
   recordings; `train.py` refuses to export when the recordings span more than one model
   or delay (a collection that straddled a server update). No value is typed by hand.

7. **Export the leave-one-language-out weights for eot-bench.** The bench adapter scores
   each language with weights that never saw it: `default.json` for English, and
   `<lang>.json` trained on the 13 other languages for every other one.

   ```bash
   eot livekit-plugins/livekit-plugins-gradium/tools/eot/train.py export-held-out \
     --out ~/git/oss/eot-bench/eot_harness/gradium_experimental_weights
   ```

   Training is seeded: re-running it on the same recordings gives byte-identical files.
   These files never ship in the plugin.

### 4. Adding a language

1. The plugin needs no change: it sends the user's `language` as given. If the language
   was collected as `any`, remove it from `ANY_HINT_LANGUAGES` in `collect.py` and from
   `GRADIUM_ANY_HINT_LANGUAGES` in eot-bench's `eot_harness/gradium_adapter.py`.
2. If it is a benchmark language (in `LANGUAGES` in `common.py`): its stream changes with
   the hint, so move its two recordings away (`data/raw/passive/<lang>.jsonl`,
   `data/raw/flush/<lang>.jsonl`), collect it again in both modes (`--lang <lang>`), and
   run steps 5 to 7. The other languages' recordings stay valid.
3. If the benchmark has no data for it: there is nothing to train or validate on. The weights still run on it, untested. Before claiming support,
   find conversational audio labelled with the same span procedure (end-of-turn and hold
   silences) and check the fallback and the learned score on it with `train.py report`
   after adding it to `LANGUAGES`. Until then, say in the docs that the language is
   transcribed but its turn-taking numbers are unmeasured.

### 5. Validation before shipping

1. **Parity test.** The pure-Python tree evaluator must reproduce scikit-learn. Step 6
   regenerates the fixture (`tests/gradium_scorer_steps.json`: a few held-out English
   turns with the scikit-learn score at every step); then:

   ```bash
   uv run python -m pytest tests/test_plugin_gradium_scorer.py -q
   ```

2. **Operating points.** Compare `train.py report` with the previous weights. Baseline
   (`933b5345@150`, English, held out, harness grid):

   | Score | Latency @ 2% | Latency @ 5% | Latency @ 10% | Cut-offs @ 300 ms | Cut-offs @ 600 ms |
   |---|---|---|---|---|---|
   | Fallback `min(p2, p3)` | 2129 ms | 1060 ms | 658 ms | 63.9% | 12.2% |
   | Learned, `default` | 1963 ms | 874 ms | 667 ms | 35.7% | 11.7% |

3. **Re-calibrate the presets.** Thresholds are tied to the weights and to the server
   model: a threshold tuned on one score scale is wrong on another. Re-run the
   calibration and the reports of "Simulator and presets" below, and copy the tuned
   values into `_LEARNED` and `_FALLBACK` in `experimental/profiles.py`.
4. **Re-run eot-bench** with the experimental adapter and the new held-out weights; the
   English row must match `train.py report` within 20 ms at the 5% budget (the plan's
   step 6 gate).
5. **Acceptance.** Ship new weights only when, on English held out: latency at the 5%
   budget is no worse than the fallback's on the same recordings and no worse than the
   previous weights by more than 30 ms (the noise level between feature sets); each
   recalibrated preset meets its false cut-off budget in the simulator report; the parity
   test and the Gradium test files pass; and the live check (below) passes on the new
   model.

### 6. Never

- Train the `default` weights on English validation turns. English is the held-out
  language every published number relies on.
- Publish numbers computed in-sample: `all-languages.json` on any benchmark language,
  or a leave-one-language-out file on its own language.
- Reuse preset thresholds across weight sets or server models. Re-calibrate.

## Simulator and presets

`simulate.py` drives the plugin's own `signals -> scorer -> TurnMachine` over recorded
turns and reports, per profile, decision and delivery latency (mean, p50, p90), false
cut-offs (in holds, and in annotated speech), turns that never commit, and the preflight
costs. Its module docstring says what is exact and what is modelled.

```bash
# the score cache changes nothing (1,800 replays compared)
eot livekit-plugins/livekit-plugins-gradium/tools/eot/simulate.py check
# calibrate all presets with preflight off, on the 13 non-English languages
eot livekit-plugins/livekit-plugins-gradium/tools/eot/simulate.py --scorer learned tune --joint
eot livekit-plugins/livekit-plugins-gradium/tools/eot/simulate.py --scorer fallback tune --joint
# report the presets in profiles.py on English (held out) and the other languages
eot livekit-plugins/livekit-plugins-gradium/tools/eot/simulate.py --scorer learned report --cohorts
# PAUSED entry against the annotated silence onset
eot livekit-plugins/livekit-plugins-gradium/tools/eot/simulate.py --scorer learned lag --preset balanced
```

Preflight on needs live recordings, because the steps after a flush differ from the
passive stream. `collect.py --mode machine` runs the plugin's machine on the live stream
and sends its flushes for real (lockstep pacing, 160 ms lead), and `simulate.py` replays
those recordings exactly:

```bash
eot livekit-plugins/livekit-plugins-gradium/tools/eot/collect.py --mode machine \
  --preset responsive --scorer learned --tag pf-responsive --lang en --subset 500 \
  --set preflight=true
eot livekit-plugins/livekit-plugins-gradium/tools/eot/simulate.py --tag pf-responsive replay
# tune the knobs that act after the eager flush on such a recording
eot livekit-plugins/livekit-plugins-gradium/tools/eot/simulate.py --tag pf-responsive \
  tune --from-recording --eagerness responsive
```

Noisy presets are measured on 500 English turns and 20 turns per other language, with
pink or babble noise mixed in at 5, 10 and 15 dB SNR:

```bash
eot livekit-plugins/livekit-plugins-gradium/tools/eot/collect.py --mode passive \
  --noise babble --snr 10 --subset 500 --lang en
eot livekit-plugins/livekit-plugins-gradium/tools/eot/simulate.py --scorer learned noise
```

## Live check

```bash
eot livekit-plugins/livekit-plugins-gradium/tools/eot/live_check.py --per-cohort 10 \
  --other fr ja de --concurrency 4
```

Pushes 33 turns through `STTExperimental(...).stream()` at real-time pacing and checks
the event order, preflight and final agreement, timestamps on the caller's timeline, a
reconnect mid-stream, a manual flush and end of input. It also measures the flush round
trip that `simulate.py --flush-rtt` models.

## Language hints

The server rejects a setup without `json_config.language` (error 1011). The plugin and
the collector send the row's language, or `any` for the benchmark languages in
`ANY_HINT_LANGUAGES` (`collect.py`), which the server has no hint for.
