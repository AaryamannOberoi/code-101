"""Writes the report folder for the Word document: artifacts/report/.

Usage (from the backend folder):
    python scripts/export_results.py                 # in-process, no server needed
    python scripts/export_results.py --url http://localhost:8000   # against a running server
"""
import argparse
import csv
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import main  # noqa: E402
from app.pipeline import runner  # noqa: E402

REPORT = runner.OUT_DIR / 'report'
TIPI = [5, 2, 6, 3, 6, 3, 6, 2, 5, 2]
DEMO_USERS = [
    {'name': 'Demo DS', 'target_role': 'Data Scientist', 'city': 'Bengaluru', 'hours_per_week': 6,
     'skills': {'big_data': 2.0, 'maths_stats': 3.0, 'coding': 2.5, 'ai_ml': 1.5, 'dashboard_story': 2.0}, 'tipi': TIPI},
    {'name': 'Demo DA', 'target_role': 'Data Analyst', 'city': 'Mumbai', 'hours_per_week': 10,
     'skills': {'big_data': 1.5, 'maths_stats': 2.5, 'coding': 2.0, 'ai_ml': 1.5, 'dashboard_story': 3.5},
     'tipi': [4, 3, 5, 4, 5, 4, 5, 3, 4, 3]},
    {'name': 'Demo DE', 'target_role': 'Data Engineer', 'city': 'Pune', 'hours_per_week': 8,
     'skills': {'big_data': 1.5, 'maths_stats': 3.0, 'coding': 3.5, 'ai_ml': 2.0, 'dashboard_story': 2.0},
     'tipi': [3, 2, 6, 2, 5, 5, 6, 2, 6, 3]},
]


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main_(url=None):
    if url:
        import httpx
        client = httpx.Client(base_url=url, timeout=30)
    else:
        from fastapi.testclient import TestClient
        main.start_pipeline(blocking=True)
        client = TestClient(main.app)
    if not client.get('/api/health').json()['pipeline_ready']:
        sys.exit('Pipeline not ready. Put the 4 data files in data/ and run again.')

    REPORT.mkdir(parents=True, exist_ok=True)
    shutil.copy(runner.OUT_DIR / 'results.json', REPORT / 'results.json')
    ev = client.get('/api/evaluation').json()
    tr = client.get('/api/transparency').json()

    metrics = ['score', 'coverage', 'overlap', 'hours', 'traits', 'anchor', 'jds_prob']
    write_csv(REPORT / 'evaluation_table.csv', ['method'] + [f'{m}_{s}' for m in metrics for s in ('mean', 'sd')],
              [[meth] + [v for m in metrics for v in ev['summary'][meth][m]] for meth in ev['summary']])
    sens = ev['sensitivity']
    write_csv(REPORT / 'sensitivity.csv', list(sens[0]), [list(r.values()) for r in sens])
    write_csv(REPORT / 'cleaning_log.csv', ['step', 'rows'], [[r['step'], r['rows']] for r in tr['cleaning_log']])
    rw = tr['scoring']['role_weights']
    areas = list(next(iter(rw.values())))
    write_csv(REPORT / 'role_weights.csv', ['role'] + areas, [[r] + [w[a] for a in areas] for r, w in rw.items()])

    samples = []
    for u in DEMO_USERS:
        r = client.post('/api/match', json=u)
        r.raise_for_status()
        samples.append({'request': u, 'response': r.json()})
    (REPORT / 'sample_matches.json').write_text(json.dumps(samples, indent=1), encoding='utf-8')

    times = []
    for i in range(50):
        t = time.perf_counter()
        client.post('/api/match', json=DEMO_USERS[i % 3]).raise_for_status()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    p95 = times[int(round(0.95 * (len(times) - 1)))]
    mode = f'HTTP against {url}' if url else 'in-process TestClient'
    (REPORT / 'latency.txt').write_text(
        f'POST /api/match latency over 50 calls ({mode})\nmean_ms: {statistics.mean(times):.1f}\np95_ms: {p95:.1f}\n',
        encoding='utf-8')

    print(f'\nReport written to {REPORT}\n')
    print(f'{"Method":<18}{"score":>8}{"coverage":>10}{"overlap":>9}{"hours":>8}{"hike prob":>11}')
    for meth, s in ev['summary'].items():
        print(f'{meth:<18}{s["score"][0]:>8.3f}{s["coverage"][0]:>10.3f}{s["overlap"][0]:>9.3f}{s["hours"][0]:>8.3f}{s["jds_prob"][0]:>11.3f}')
    print(f'\nWins vs baselines: {ev["wins_pct"]}')
    print(f'Solo hike prob: {ev["solo_hike_prob"]}   Train vs test profile corr: {ev["profile_train_test_corr"]}')
    print(f'Match latency: mean {statistics.mean(times):.1f} ms, p95 {p95:.1f} ms')
    for s in samples:
        g = s['response']['group']
        print(f'{s["request"]["target_role"]:<16} group score {g["score"]}, members '
              f'{", ".join(m["name"] for m in g["members"])}; project: {s["response"]["project"]["title"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', help='Base URL of a running server (default: run in-process)')
    main_(ap.parse_args().url)
