"""Onboarding: TIPI scoring, persona assignment, weighted skill gap, simulated pool."""
import numpy as np

from .models import SF
from .profiles import AREA_KEYS, AREA_LABEL
from .matcher import SK, TR, simulate_students

TIPI_ITEMS = [
    'Extraverted, enthusiastic', 'Critical, quarrelsome', 'Dependable, self-disciplined', 'Anxious, easily upset',
    'Open to new experiences, complex', 'Reserved, quiet', 'Sympathetic, warm', 'Disorganized, careless',
    'Calm, emotionally stable', 'Conventional, uncreative',
]

FIRST_NAMES = [
    'Aarav', 'Riya', 'Kabir', 'Ananya', 'Vihaan', 'Diya', 'Arjun', 'Ishita', 'Reyansh', 'Saanvi',
    'Aditya', 'Meera', 'Krishna', 'Aisha', 'Rohan', 'Tara', 'Dev', 'Kavya', 'Nikhil', 'Pooja',
    'Siddharth', 'Neha', 'Yash', 'Sneha', 'Karan', 'Priya', 'Aryan', 'Nisha', 'Rahul', 'Anika',
    'Varun', 'Zoya', 'Ishaan', 'Lavanya', 'Harsh', 'Sana', 'Manav', 'Shruti', 'Kunal', 'Aditi',
    'Pranav', 'Mira', 'Tanmay', 'Jiya', 'Rudra', 'Simran', 'Om', 'Divya', 'Atharv', 'Navya',
    'Sahil', 'Rhea', 'Aniket', 'Kiara', 'Parth', 'Avni', 'Dhruv', 'Myra', 'Ayaan', 'Tanvi',
    'Vivaan', 'Sara', 'Neel', 'Pihu', 'Advait', 'Ira', 'Shaurya', 'Khushi', 'Laksh', 'Nandini',
    'Samar', 'Prisha', 'Veer', 'Anvi', 'Rishi',
]
INITIALS = 'ABCDGJKMNPRSTV'
MENTOR_POOL = {'Data Scientist': 4, 'Data Analyst': 5, 'Business Analyst': 3, 'Data Engineer': 3,
               'Machine Learning Engineer': 2, 'Data Architect': 1}


def tipi_to_sds(tipi):
    """10 TIPI answers (1..7) -> Big Five on the SDS scale (approx 17..68), in SF order."""
    x = [float(v) for v in tipi]
    r = lambda v: 8 - v
    extraversion = (x[0] + r(x[5])) / 2
    agreeableness = (r(x[1]) + x[6]) / 2
    conscientiousness = (x[2] + r(x[7])) / 2
    stability = (r(x[3]) + x[8]) / 2
    openness = (x[4] + r(x[9])) / 2
    neuroticism = 8 - stability
    raw = {'neuroticism': neuroticism, 'extraversion': extraversion, 'openness_to_experience': openness,
           'agreeableness': agreeableness, 'conscientiousness': conscientiousness}
    return {k: int(17 + (v - 1) / 6 * 51 + 0.5) for k, v in raw.items()}


def learner_persona(sk, jprof, names):
    """Shape, not level: centre both vectors on their own mean, nearest centroid by cosine."""
    v = np.asarray(sk, float)
    v = v - v.mean()
    nv = np.linalg.norm(v)
    if nv < 1e-9:  # perfectly flat profile reads as even, mid-level skills
        return 'Foundation Builder'
    best, best_cos = None, -2
    for cid, row in jprof.iterrows():
        c = row.values - row.values.mean()
        cos = float(np.dot(v, c) / (nv * np.linalg.norm(c) + 1e-9))
        if cos > best_cos:
            best, best_cos = names[int(cid)], cos
    return best


def workstyle_persona(tr, skm, sksc, names):
    cid = int(skm.predict(sksc.transform(np.atleast_2d(np.asarray(tr, float))))[0])
    return names[cid]


def weighted_gap(skills, W):
    return [{'area': a, 'label': AREA_LABEL[a], 'your_level': round(float(skills[a]), 2), 'role_weight': round(float(W[i]), 3),
             'weighted_gap': round(float(W[i] * (5 - skills[a]) / 4), 3)} for i, a in enumerate(AREA_KEYS)]


def build_pool(j, s, role_share, cities, city_p, engine, jprof, jnames, skm, sksc, snames, n=150, seed=2026):
    """150 simulated students (skills and traits identical to the reference seed-2026 example), plus names and cities."""
    pool = simulate_students(n, seed, j, s, role_share)
    g = np.random.default_rng(seed + 1)
    combos = [f'{f} {i}.' for f in FIRST_NAMES for i in INITIALS]
    picks = g.choice(len(combos), n, replace=False)
    pool['name'] = [combos[k] for k in picks]
    pool['city'] = g.choice(cities, n, p=city_p)
    pool['trait_success_prob'] = engine.success_prob(pool[TR].values)
    pool['learner_persona'] = [learner_persona(r, jprof, jnames) for r in pool[SK].values]
    pool['workstyle_persona'] = [workstyle_persona(r, skm, sksc, snames) for r in pool[TR].values]
    return pool
