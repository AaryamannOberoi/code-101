"""Explanation, 8-week learning order, shared project picker."""
import numpy as np

from .profiles import AREA_KEYS, AREA_LABEL

GOALS = {
    'big_data': ['Spark basics on a sample dataset', 'Hive and SQL on a data lake table', 'Build a small batch ETL job'],
    'maths_stats': ['Hypothesis tests and regression', 'Probability and sampling refresher', 'Time series and forecasting basics'],
    'coding': ['Python data wrangling with pandas', 'SQL joins and window functions', 'Write clean, reusable functions and tests'],
    'ai_ml': ['Supervised learning basics', 'Train a first classifier in scikit-learn', 'Model evaluation: AUC, cross-validation'],
    'dashboard_story': ['Build a dashboard, tell one story with it', 'Chart choice and design basics', 'Present findings in a one-page memo'],
}
MIN_PLAN_WEIGHT = 0.03  # areas below this role weight are negligible for the plan

ROLE_CODE = {'DA': 'Data Analyst', 'BA': 'Business Analyst', 'DS': 'Data Scientist', 'DE': 'Data Engineer',
             'MLE': 'Machine Learning Engineer', 'DArch': 'Data Architect'}
PROJECTS = [
    ('Hiring demand dashboard from job postings', ['dashboard_story', 'coding', 'maths_stats'], ['DA', 'BA']),
    ('Customer churn predictor with an explainer dashboard', ['ai_ml', 'maths_stats', 'coding', 'dashboard_story'], ['DS', 'DA']),
    ('Resume to job-skill matcher (NLP)', ['ai_ml', 'coding'], ['DS', 'MLE']),
    ('Streaming ETL pipeline on Spark', ['big_data', 'coding'], ['DE', 'DArch']),
    ('Sales forecasting with time series', ['maths_stats', 'coding', 'dashboard_story'], ['DA', 'DS']),
    ('A/B test analysis and decision memo', ['maths_stats', 'dashboard_story'], ['DA', 'BA']),
    ('Campus data warehouse with a reporting layer', ['big_data', 'coding', 'dashboard_story'], ['DE', 'DArch', 'BA']),
    ('Image classifier with a CNN', ['ai_ml', 'coding'], ['MLE', 'DS']),
    ('Fraud detection on imbalanced data', ['ai_ml', 'maths_stats', 'coding'], ['DS', 'MLE']),
    ('Requirements to dashboard: a BA case study', ['dashboard_story', 'maths_stats'], ['BA']),
    ('Recommendation engine on Spark', ['ai_ml', 'big_data', 'coding'], ['MLE', 'DE', 'DS']),
    ('City job market data story', ['dashboard_story', 'maths_stats', 'coding'], ['DA', 'BA']),
]


def _fmt_list(items):
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def explanation(members, user_gap, parts, lead_name, lead_style):
    """members: list of dicts with name, is_user, skills, hours_per_week. Returns (sentences, uncovered_areas)."""
    top2 = [g['area'] for g in sorted(user_gap, key=lambda g: -g['weighted_gap'])[:2]]
    others = [m for m in members if not m['is_user']]
    user = next(m for m in members if m['is_user'])
    sentences = []
    covers = {}  # member name -> [(area, level)]
    for a in top2:
        best = max(others, key=lambda m: m['skills'][a]) if others else None
        if best is None or best['skills'][a] <= user['skills'][a]:
            sentences.append(f'Nobody in the group is ahead of you in {AREA_LABEL[a]}, so plan extra practice there')
            continue
        covers.setdefault(best['name'], []).append((a, best['skills'][a]))
    for name, items in covers.items():
        desc = _fmt_list([f'{AREA_LABEL[a]} ({lvl:.2f})' for a, lvl in items])
        if len(items) == 2:
            tail = 'your two biggest gaps'
        elif items[0][0] == top2[0]:
            tail = 'your biggest gap'
        else:
            tail = 'your second biggest gap'
        sentences.append(f'{name} leads {desc}, {tail}')
    ov = parts['overlap']
    if ov < 0.45:
        sentences.append('Low overlap: the group members are strong in different areas')
    elif ov < 0.6:
        sentences.append('Moderate overlap: a few shared strengths, but mostly different ones')
    else:
        sentences.append('High overlap: the group shares similar strengths, so expect to learn some areas together')
    hours = [m['hours_per_week'] for m in members]
    n = len(members)
    if max(hours) == min(hours):
        sentences.append(f'All {n} of you can give {hours[0]} hours a week, so sessions line up')
    else:
        sentences.append(f'Weekly hours range from {min(hours)} to {max(hours)}, so sessions are planned around {min(hours)} hours')
    who = 'You are' if lead_name == user['name'] else f'{lead_name} is'
    sentences.append(f'{who} the accountability lead this week ({lead_style} work style); the role rotates weekly')
    group_max = {a: max(m['skills'][a] for m in members) for a in AREA_KEYS}
    uncovered = [a for a in AREA_KEYS if group_max[a] < 3.0]
    return sentences[:5], uncovered


def group_gap(members):
    return {a: float(np.mean([(5 - m['skills'][a]) / 4 for m in members])) for a in AREA_KEYS}


def learning_plan(members, W):
    gg = group_gap(members)
    w = dict(zip(AREA_KEYS, W))
    ranked = sorted(AREA_KEYS, key=lambda a: -w[a] * gg[a])
    eligible = [a for a in ranked if w[a] >= MIN_PLAN_WEIGHT] or ranked[:1]
    slots = [eligible[0]] + eligible[:5]
    i = 1
    while len(slots) < 6:  # fewer than 5 relevant areas: give spare weeks to the top areas again
        slots.append(eligible[i % len(eligible)])
        i += 1
    hours = min(m['hours_per_week'] for m in members)
    seen = {}
    weeks = []
    for wk, a in enumerate(slots, start=1):
        k = seen.get(a, 0)
        seen[a] = k + 1
        goals = GOALS[a][:2] if k == 0 else [GOALS[a][min(1 + k, len(GOALS[a]) - 1)]]
        lead = max(members, key=lambda m: m['skills'][a])['name']
        weeks.append({'week': wk, 'focus_area': a, 'lead': lead, 'hours': hours, 'goals': goals})
    weeks.append({'week': 7, 'focus_area': 'project', 'lead': 'All', 'hours': hours, 'goals': ['Shared project build']})
    weeks.append({'week': 8, 'focus_area': 'project', 'lead': 'All', 'hours': hours, 'goals': ['Shared project demo and write-up']})
    return {'weeks': weeks}


def pick_project(members, W, role):
    gg = group_gap(members)
    w = dict(zip(AREA_KEYS, W))
    total = sum(w[a] * gg[a] for a in AREA_KEYS) or 1e-9
    best = None
    for title, areas, roles in PROJECTS:
        pct = sum(w[a] * gg[a] for a in areas) / total * 100
        key = (round(pct, 6), role in [ROLE_CODE[r] for r in roles])
        if best is None or key > best[0]:
            best = (key, title, areas, pct)
    _, title, areas, pct = best
    top2 = sorted(AREA_KEYS, key=lambda a: -w[a] * gg[a])[:2]
    hit = [a for a in top2 if a in areas]
    if len(hit) == 2:
        why = f"Covers {len(areas)} of 5 areas and the group's top two gaps"
    elif hit:
        why = f"Covers {len(areas)} of 5 areas including the group's biggest gap ({AREA_LABEL[top2[0]]})" if hit[0] == top2[0] \
            else f"Covers {len(areas)} of 5 areas including {AREA_LABEL[hit[0]]}, one of the group's top two gaps"
    else:
        why = f'Covers {len(areas)} of 5 areas that matter for the role'
    return {'title': title, 'areas': list(areas), 'gap_closed_pct': round(pct, 1), 'why': why}
