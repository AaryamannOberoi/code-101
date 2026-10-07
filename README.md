# SkillCircle API

Team CODE 101, SAS CU Hackathon 2026. A goal-based cohort builder: FastAPI backend that ports
`reference/reference_analysis.py` (seed 42) and serves the shapes in `reference/mock_api.json`.
Everything runs locally: no external APIs, no LLM calls, no cloud database.

## Setup

```
pip install -r requirements.txt
```

Put the four input files in `data/`:

```
data/Analytics_Jobs.csv
data/DataScience_Jobs.csv
data/JDS_Skill_Traits.xlsx
data/SDS_Personality_Traits.xlsx
```

## Run

```
uvicorn app.main:app --reload --port 8000
```

- With no data files, every GET route serves `mock_api.json` and `POST /api/match` returns the mock response, so the frontend can work right away. `/api/health` shows `"mode": "mock"` and lists the missing files.
- With data files and no cache, the server starts at once (mock mode) and builds the pipeline in a background thread (modules A to E, 30 simulation runs). When it finishes, `/api/health` shows `"pipeline_ready": true` and all routes switch to live numbers.
- The build writes `artifacts/results.json`, `artifacts/*.csv` and `artifacts/pipeline.pkl`. Later starts load the cache in about a second. The cache is rebuilt automatically if a data file changes.

To build the pipeline in the foreground (for example before a demo):

```
python -c "from app.pipeline import runner; runner.load_or_build(force=True)"
```

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | status, pipeline_ready, built_at (plus mode, missing_data_files) |
| GET | `/api/meta` | team, areas, roles, cities, dataset_stats |
| GET | `/api/insights` | role landscape, tests, area profiles, top skills, rules, area vs salary, models, personas |
| GET | `/api/evaluation` | simulation summary, wins, Wilcoxon p, sensitivity, solo hike prob, train vs test correlation |
| GET | `/api/transparency` | cleaning log, data issues, scoring weights and formula, models, assumptions, limitations |
| POST | `/api/match` | user personas and gap, group of 3, explanation, 8-week plan, project, community |
| GET | `/api/pool?role=` | simulated pool students (seed 2026) |
| GET | `/api/export` | full results.json download |

Interactive docs: http://localhost:8000/docs

Invalid input to `/api/match` (unknown role, skills outside 1 to 5, hours outside 1 to 40, tipi not exactly 10 integers from 1 to 7) returns 422 with a `message` field that names each bad field.

## Tests

```
python -m pytest -q
```

Checks every route against the mock shape (same keys, nesting and types), a match round-trip, that the stored pool is not changed by a match, 422 on bad input, and no long dashes in any response. With data present the tests run on live numbers.

## Results export (for the Word report)

```
python scripts/export_results.py                               # in-process
python scripts/export_results.py --url http://localhost:8000   # against a running server
```

Writes `artifacts/report/`: results.json, evaluation_table.csv, sensitivity.csv, cleaning_log.csv,
role_weights.csv, sample_matches.json (3 demo users), latency.txt (mean and p95 of 50 match calls), and prints a summary table.

## Layout

```
app/main.py                 FastAPI app, routes, mock fallback
app/service.py              pipeline state to API payloads
app/schemas.py              request and response models for /api/match
app/pipeline/clean.py       Module A, Module B cleaning and role mapping
app/pipeline/profiles.py    skill areas, role profiles, lift, rules, chi-square
app/pipeline/models.py      JDS and SDS tests, Wald logistic, CV models, personas
app/pipeline/matcher.py     weights, group score, greedy matcher, baselines, simulation
app/pipeline/onboarding.py  TIPI scoring, personas, gap, simulated pool
app/pipeline/planner.py     explanation, 8-week plan, project picker
app/pipeline/runner.py      runs everything once, writes artifacts, caches
```

## Design notes

- Matching formula: `score = 0.45 x coverage + 0.20 x (1 - overlap) + 0.15 x hours_fit + 0.20 x trait_score`.
- The simulation uses the reference formulas and tie-breaking, on numpy arrays, so a full build takes seconds, not minutes.
- Persona names are matched to k-means clusters by nearest centroid (seed-42 centroids), so names stay right if cluster ids shift between library versions.
- Learner persona: the skill vector and each centroid are centred on their own mean (shape, not level), then the nearest by cosine. Work style: SDS scaler, then nearest centroid.
- The user is placed into a copy of the pool's arrays and used as the forced seed of a greedy group, so the stored pool never changes.
- `groups_in_city` counts pool groups (all roles) where at least 2 members live in the user's city.
- If a role has fewer than 2 pool students, the match borrows peers from the whole pool.
