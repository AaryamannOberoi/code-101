"""Runs the full pipeline once, writes artifacts, caches everything to artifacts/pipeline.pkl."""
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import clean, matcher, models, onboarding, profiles

ROOT = Path(__file__).resolve().parents[2]
# env overrides exist only for smoke tests on synthetic data
DATA_DIR = Path(os.environ.get('SKILLCIRCLE_DATA_DIR', ROOT / 'data'))
OUT_DIR = Path(os.environ.get('SKILLCIRCLE_OUT_DIR', ROOT / 'artifacts'))
CACHE = OUT_DIR / 'pipeline.pkl'
DATA_FILES = ['Analytics_Jobs.csv', 'DataScience_Jobs.csv', 'JDS_Skill_Traits.xlsx', 'SDS_Personality_Traits.xlsx']
PIPELINE_VERSION = 1


def data_present():
    return all((DATA_DIR / f).exists() for f in DATA_FILES)


def missing_files():
    return [f for f in DATA_FILES if not (DATA_DIR / f).exists()]


def _fingerprint():
    return [PIPELINE_VERSION] + [((DATA_DIR / f).stat().st_size, int((DATA_DIR / f).stat().st_mtime)) for f in DATA_FILES]


def conv(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [conv(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(x) for x in o]
    return o


def build(log=print):
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    R = {}
    rng = np.random.default_rng(42)

    log('Module A: DataScience_Jobs')
    R['A'] = clean.module_a(DATA_DIR, OUT_DIR)

    log('Module B: Analytics_Jobs')
    core, B = clean.module_b_clean(DATA_DIR)
    core, prof_tr, prof_te, prof_all = profiles.module_b_analysis(core, B, rng, OUT_DIR)
    R['B'] = B

    log('Modules C and D: JDS and SDS models')
    C, Dm, M = models.modules_cd(DATA_DIR)
    R['C'], R['D'] = C, Dm

    log('Module E: matching engine')
    engine = matcher.Engine(M['jm'], M['jsc'], M['sm'], M['ssc'])
    payoff = matcher.payoff_weights(M['jm'])
    W_tr = matcher.role_weights(prof_tr, payoff)
    W_te = matcher.role_weights(prof_te, payoff)
    role_share = core.role.value_counts(normalize=True)
    E = {'payoff_weights': payoff,
         'role_weights_train': {r: dict(zip(profiles.AREA_KEYS, np.round(w, 3))) for r, w in W_tr.items()},
         'role_share': role_share.round(3).to_dict()}
    sims = matcher.run_simulation(engine, M['j'], M['s'], role_share, W_tr, W_te, log=log)
    sims.to_csv(OUT_DIR / 'E_sims.csv', index=False)
    matcher.summarise(sims, E)
    E['n_students'], E['n_sims'], E['weights'] = 150, 30, dict(matcher.BASE_WEIGHTS)
    log('  sensitivity runs')
    E['sensitivity'] = matcher.run_sensitivity(engine, M['j'], M['s'], role_share, W_tr, W_te)
    E['example'] = matcher.worked_example(engine, M['j'], M['s'], role_share, W_tr)
    R['E'] = E
    R['AREA_LABEL'] = profiles.AREA_LABEL
    R = conv(R)
    with open(OUT_DIR / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(R, f, indent=1, default=str)
    M['jd'].to_csv(OUT_DIR / 'C_jds_personas.csv', index=False)
    M['sd'].to_csv(OUT_DIR / 'D_sds_personas.csv', index=False)

    log('Onboarding pool (seed 2026)')
    top_cities = B['top_cities']
    cities = list(top_cities)
    city_p = np.array(list(top_cities.values()), float)
    city_p /= city_p.sum()
    jnames = C['persona_names']
    snames = Dm['persona_names']
    pool = onboarding.build_pool(M['j'], M['s'], role_share, cities, city_p, engine, M['jprof'], jnames,
                                 M['skm'], M['sksc'], snames)
    SKa, Ha, TPa = matcher.arrays(pool, engine)
    pool_groups = {role: engine.greedy(SKa, Ha, TPa, idx, W_tr[role]) if len(idx) >= 2 else [idx]
                   for role, idx in matcher.role_positions(pool).items()}

    state = {
        'fingerprint': _fingerprint(),
        'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'build_seconds': round(time.time() - t0, 1),
        'R': R,
        'engine': engine,
        'W_tr': W_tr,
        'jprof': M['jprof'], 'jnames': jnames, 'skm': M['skm'], 'sksc': M['sksc'], 'snames': snames,
        'pool': pool,
        'pool_groups': conv(pool_groups),
        'city_postings': core.city.str.lower().value_counts().to_dict(),
        'cities': cities,
    }
    with open(CACHE, 'wb') as f:
        pickle.dump(state, f)
    log(f'Pipeline built in {state["build_seconds"]} s')
    return state


def load_or_build(log=print, force=False):
    """Returns the pipeline state, or None if the data files are missing."""
    if not data_present():
        return None
    if CACHE.exists() and not force:
        try:
            with open(CACHE, 'rb') as f:
                state = pickle.load(f)
            if state.get('fingerprint') == _fingerprint():
                return state
            log('Cache is stale (data or pipeline version changed), rebuilding')
        except Exception as e:  # corrupt or incompatible cache
            log(f'Cache unreadable ({e}), rebuilding')
    return build(log)
