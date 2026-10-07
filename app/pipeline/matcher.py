"""Module E: payoff and role weights, group scoring, greedy matcher, baselines, simulation.

Same formulas and tie-breaking as reference_analysis.py. Students are held as numpy arrays
(positions instead of DataFrame labels) so the 30-run simulation stays fast.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .models import JF, SF
from .profiles import AREA_KEYS

BASE_WEIGHTS = {'coverage': 0.45, 'overlap': 0.20, 'hours': 0.15, 'traits': 0.20}
SENSITIVITY = {
    'Base': dict(BASE_WEIGHTS),
    'Coverage heavy': {'coverage': .6, 'overlap': .15, 'hours': .1, 'traits': .15},
    'Equal weights': {'coverage': .25, 'overlap': .25, 'hours': .25, 'traits': .25},
    'No traits': {'coverage': .55, 'overlap': .25, 'hours': .2, 'traits': 0},
}
FORMULA = 'score = 0.45 x coverage + 0.20 x (1 - overlap) + 0.15 x hours_fit + 0.20 x trait_score'
SK = [f'sk_{i}' for i in range(5)]
TR = [f'tr_{i}' for i in range(5)]
METHODS = ['Our matcher', 'Random grouping', 'Similarity-only']
METRICS = ['coverage', 'overlap', 'hours', 'traits', 'anchor', 'score', 'jds_prob']


def payoff_weights(jm):
    coef = dict(zip(JF, jm.coef_[0]))
    area_from_jds = dict(zip(JF, AREA_KEYS))
    payoff = {area_from_jds[f]: max(coef[f], 0) for f in JF}
    mx = max(payoff.values()) or 1
    return {k: 0.5 + 0.5 * v / mx for k, v in payoff.items()}  # 0.5..1


def role_weights(prof, payoff):
    W = {}
    for role, row in prof.iterrows():
        w = np.array([row[ar] * payoff[ar] for ar in AREA_KEYS]) + 1e-6
        W[role] = w / w.sum()
    return W


def simulate_students(n, seed, j, s, role_share):
    g = np.random.default_rng(seed)
    js = j[JF].values[g.integers(0, len(j), n)]
    # students are earlier in their journey: shift down by U(0.5,1.5), jitter, clip to 1..5
    skills = np.clip(js - g.uniform(0.5, 1.5, (n, 1)) + g.normal(0, 0.4, js.shape), 1, 5)
    tr_ = s[SF].values[g.integers(0, len(s), n)] + g.normal(0, 2, (n, 5))
    hours = g.choice([4, 6, 8, 10, 12, 15], n, p=[.15, .2, .25, .2, .12, .08])
    roles = g.choice(role_share.index, n, p=role_share.values)
    return pd.DataFrame({'role': roles, 'hours': hours, **{f'sk_{i}': skills[:, i] for i in range(5)},
                         **{f'tr_{i}': tr_[:, i] for i in range(5)}})


def sizes(n):
    if n < 2:
        return [n]
    k = n // 3
    r = n % 3
    if r == 0:
        return [3] * k
    if r == 1:
        return [3] * (k - 1) + [4]
    return [3] * k + [2]


class Engine:
    """Holds the fitted JDS and SDS logistic models and scores groups."""

    def __init__(self, jm, jsc, sm, ssc):
        self.jm, self.jsc, self.sm, self.ssc = jm, jsc, sm, ssc

    def success_prob(self, tr_vals):
        return self.sm.predict_proba(self.ssc.transform(np.atleast_2d(tr_vals)))[:, 1]

    def hike_prob(self, sk_vals):
        return self.jm.predict_proba(self.jsc.transform(np.atleast_2d(sk_vals)))[:, 1]

    # ---- group scoring on arrays: sk (n,5), hours (n,), tp (n,) ----
    @staticmethod
    def parts(sk, hours, tp, W):
        covered = sk.max(axis=0)
        coverage = float(np.dot(W, covered / 5))
        n = len(sk)
        if n > 1:
            c = sk - sk.mean(axis=1, keepdims=True)
            sims = [np.dot(c[i], c[k]) / (np.linalg.norm(c[i]) * np.linalg.norm(c[k]) + 1e-9)
                    for i, k in itertools.combinations(range(n), 2)]
            overlap = (np.mean(sims) + 1) / 2
        else:
            overlap = 1.0
        h_fit = 1 - (hours.max() - hours.min()) / 11
        traits = float(0.7 * tp.max() + 0.3 * tp.mean())
        return {'coverage': coverage, 'overlap': float(overlap), 'hours': float(h_fit), 'traits': traits}

    @staticmethod
    def score_parts(p, wts=BASE_WEIGHTS):
        return (wts['coverage'] * p['coverage'] + wts['overlap'] * (1 - p['overlap'])
                + wts['hours'] * p['hours'] + wts['traits'] * p['traits'])

    def score(self, sk, hours, tp, W, wts=BASE_WEIGHTS):
        return self.score_parts(self.parts(sk, hours, tp, W), wts)

    # ---- matchers: take global arrays plus the list of member positions for one role ----
    def greedy(self, SKa, Ha, TPa, idx, W, wts=BASE_WEIGHTS, forced_seed=None, first_only=False):
        idx = list(idx)
        pool = list(idx)
        groups = []
        gap = ((5 - SKa[idx]) / 4 * W).sum(axis=1)
        order = [idx[i] for i in np.argsort(-gap)]
        for gi, sz in enumerate(sizes(len(pool))):
            if gi == 0 and forced_seed is not None:
                seed = forced_seed
            else:
                seed = next(i for i in order if i in pool)
            G = [seed]
            pool.remove(seed)
            while len(G) < sz:
                best = max(pool, key=lambda c: self.score(SKa[G + [c]], Ha[G + [c]], TPa[G + [c]], W, wts))
                G.append(best)
                pool.remove(best)
            groups.append(G)
            if first_only:
                break
        return groups

    @staticmethod
    def random_match(idx, seed):
        perm = list(np.random.default_rng(seed).permutation(np.asarray(idx)))
        out, p = [], 0
        for sz in sizes(len(perm)):
            out.append(perm[p:p + sz])
            p += sz
        return out

    @staticmethod
    def similarity_match(SKa, idx):
        pool = list(idx)
        out = []
        for sz in sizes(len(pool)):
            seed = pool[0]
            pool.remove(seed)
            G = [seed]
            v = SKa[seed].astype(float)
            while len(G) < sz:
                best = min(pool, key=lambda c: np.linalg.norm(SKa[c].astype(float) - v))
                G.append(best)
                pool.remove(best)
            out.append(G)
        return out

    def evaluate(self, groups_by_role, SKa, Ha, TPa, W, wts=BASE_WEIGHTS):
        rows, best_vecs = [], []
        for role, groups in groups_by_role.items():
            for G in groups:
                p = self.parts(SKa[G], Ha[G], TPa[G], W[role])
                rows.append({**p, 'anchor': float(TPa[G].max() >= 0.5), 'score': self.score_parts(p, wts), 'size': len(G)})
                best_vecs.append(SKa[G].max(axis=0))
        ev = pd.DataFrame(rows)
        ev['jds_prob'] = self.hike_prob(np.array(best_vecs))
        return ev


def arrays(st, engine):
    SKa = st[SK].values.astype(float)
    Ha = st.hours.values.astype(float)
    TPa = engine.success_prob(st[TR].values)
    return SKa, Ha, TPa


def role_positions(st):
    return {role: list(np.flatnonzero(st.role.values == role)) for role in sorted(st.role.unique())}


def run_simulation(engine, j, s, role_share, W_tr, W_te, n_stud=150, n_sim=30, log=print):
    sims = []
    for sim in range(n_sim):
        st = simulate_students(n_stud, 1000 + sim, j, s, role_share)
        SKa, Ha, TPa = arrays(st, engine)
        res = {}
        for method in METHODS:
            gb = {}
            for role, idx in role_positions(st).items():
                if len(idx) < 2:
                    continue
                if method == 'Our matcher':
                    gb[role] = engine.greedy(SKa, Ha, TPa, idx, W_tr[role])
                elif method == 'Random grouping':
                    gb[role] = engine.random_match(idx, sim)
                else:
                    gb[role] = engine.similarity_match(SKa, idx)
            ev = engine.evaluate(gb, SKa, Ha, TPa, W_te)  # judged on held-out posting weights
            res[method] = ev.mean(numeric_only=True)
        solo = engine.hike_prob(SKa).mean()
        for m_, v in res.items():
            sims.append({'sim': sim, 'method': m_, **v.to_dict(), 'solo_prob': solo})
        log(f'  simulation {sim + 1}/{n_sim} done')
    return pd.DataFrame(sims)


def summarise(sims, E):
    summ = sims.groupby('method')[METRICS].agg(['mean', 'std']).round(4)
    E['summary'] = {m_: {k: [float(summ.loc[m_, (k, 'mean')]), float(summ.loc[m_, (k, 'std')])] for k in METRICS}
                    for m_ in summ.index}
    E['solo_prob_mean'] = float(sims.solo_prob.mean())
    piv = sims.pivot(index='sim', columns='method')
    tests = {}
    for base in ['Random grouping', 'Similarity-only']:
        for k in ['score', 'coverage', 'jds_prob', 'overlap', 'hours', 'anchor']:
            try:
                tests[f'{base}|{k}'] = float(stats.wilcoxon(piv[(k, 'Our matcher')], piv[(k, base)]).pvalue)
            except ValueError:  # all differences zero
                tests[f'{base}|{k}'] = 1.0
    E['wilcoxon'] = tests
    E['wins_pct'] = {base: float((piv[('score', 'Our matcher')] > piv[('score', base)]).mean() * 100)
                     for base in ['Random grouping', 'Similarity-only']}


def run_sensitivity(engine, j, s, role_share, W_tr, W_te, n_stud=150):
    sens = []
    st = simulate_students(n_stud, 777, j, s, role_share)
    SKa, Ha, TPa = arrays(st, engine)
    for name, wts in SENSITIVITY.items():
        gb = {role: engine.greedy(SKa, Ha, TPa, idx, W_tr[role], wts) for role, idx in role_positions(st).items() if len(idx) >= 2}
        ev = engine.evaluate(gb, SKa, Ha, TPa, W_te, wts)
        sens.append({'setting': name, 'coverage': ev.coverage.mean(), 'jds_prob': ev.jds_prob.mean(), 'traits': ev.traits.mean(),
                     'anchor': ev.anchor.mean(), 'hours': ev.hours.mean(), 'overlap': ev.overlap.mean()})
    return pd.DataFrame(sens).round(4).to_dict('records')


def worked_example(engine, j, s, role_share, W_tr, n_stud=150):
    st = simulate_students(n_stud, 2026, j, s, role_share)
    SKa, Ha, TPa = arrays(st, engine)
    idx = role_positions(st)['Data Scientist']
    G = engine.greedy(SKa, Ha, TPa, idx, W_tr['Data Scientist'])[0]
    W = W_tr['Data Scientist']
    parts = engine.parts(SKa[G], Ha[G], TPa[G], W)
    return {'members': st.iloc[G][SK + TR + ['hours']].round(2).to_dict('records'), 'parts': parts,
            'score': engine.score_parts(parts), 'jds_prob': float(engine.hike_prob(SKa[G].max(axis=0))[0]),
            'solo_probs': engine.hike_prob(SKa[G]).round(3).tolist(), 'trait_probs': TPa[G].round(3).tolist(),
            'weights': dict(zip(AREA_KEYS, np.round(W, 3)))}
