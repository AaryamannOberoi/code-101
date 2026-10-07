"""Turns the cached pipeline state into API payloads shaped exactly like mock_api.json."""
import json
from pathlib import Path

import numpy as np

from .pipeline.matcher import FORMULA, SK, TR
from .pipeline.models import JF, LEARNER_PERSONAS, SF, WORKSTYLE_PERSONAS
from .pipeline.onboarding import MENTOR_POOL, learner_persona, tipi_to_sds, weighted_gap, workstyle_persona
from .pipeline.planner import explanation, learning_plan, pick_project
from .pipeline.profiles import AREA_KEYS, AREA_LABEL

MOCK = json.loads((Path(__file__).parent / 'mock_api.json').read_text(encoding='utf-8'))
ROLES = ['Business Analyst', 'Data Analyst', 'Data Architect', 'Data Engineer', 'Data Scientist', 'Machine Learning Engineer']
NOISE_RULE_ITEMS = {'business process', 'outsourcing'}
CITY_ALIAS = {'bangalore': 'bengaluru', 'gurugram': 'gurgaon', 'bombay': 'mumbai', 'madras': 'chennai', 'new delhi': 'delhi'}
CEILING_LABEL = {'big_data_skills': 'big data', 'maths_stats_skills': 'maths and stats', 'coding_skills': 'coding',
                 'ai_and_ml_skills': 'AI and ML', 'dashboard_and_storytelling_skills': 'storytelling'}
JF_TO_AREA = dict(zip(JF, AREA_KEYS))


def _r(x, n):
    return round(float(x), n)


# ---------------- GET views ----------------
def meta(st):
    R = st['R']
    return {
        'team': 'CODE 101',
        'event': 'SAS CU Hackathon 2026',
        'areas': [{'key': a, 'label': AREA_LABEL[a]} for a in AREA_KEYS],
        'roles': ROLES,
        'cities': st['cities'],
        'dataset_stats': {
            'analytics_jobs_raw': R['B']['rows_raw'],
            'analytics_jobs_core': R['B']['cleaning_log'][-1][1],
            'ds_jobs_rows': R['A']['rows_raw'],
            'ds_jobs_companies': R['A']['companies'],
            'jds_rows': R['C']['n'],
            'sds_rows': R['D']['n'],
        },
    }


def _model_block(M):
    return {'ttests': M['ttests'], 'logit': M['logit'], 'models': M['models'], 'rf_importance': M['rf_importance'],
            'mcfadden_r2': _r(M['mcfadden_r2'], 3)}


def _personas(M, catalogue, feats, keymap, rate_key, rate_name):
    desc = {n: d for n, d, _ in catalogue}
    out = []
    for cid in sorted(M['persona_profile'], key=int):
        name = M['persona_names'][cid]
        out.append({'id': int(cid), 'name': name, 'description': desc[name],
                    'profile': {keymap.get(f, f): M['persona_profile'][cid][f] for f in feats},
                    'size': int(M['persona_size'][cid]), rate_name: M[rate_key][cid]})
    return out


def insights(st):
    R = st['R']
    A, B, C, D = R['A'], R['B'], R['C'], R['D']
    rules = [r for r in B['rules'] if not ({r['if'], r['then']} & NOISE_RULE_ITEMS)]
    return {
        'role_landscape': [{'role': r['job_title'], 'listings': int(r['listings']), 'median_salary_lpa': r['median_avg_salary'],
                            'mean_min_experience': r['mean_min_exp'], 'attractiveness': r['attractiveness']} for r in A['role_table']],
        'role_tests': {'anova_F': _r(A['anova_F'], 1), 'anova_p': A['anova_p'], 'kruskal_H': _r(A['kruskal_H'], 1),
                       'spearman_experience_salary': _r(A['spearman_exp_salary'][0], 3)},
        'role_area_profile': B['area_profile'],
        'top_skills': B['top_skills'],
        'skill_rules': rules,
        'area_vs_salary': {a: {'cramers_v': _r(v['cramers_v'], 3), 'p': v['p'], 'pct_high_band_with': _r(v['high_band_with'], 1),
                               'pct_high_band_without': _r(v['high_band_without'], 1)} for a, v in B['chi_area_salary'].items()},
        'skills_that_pay': _model_block(C),
        'traits_that_succeed': _model_block(D),
        'personas': {
            'learner': _personas(C, LEARNER_PERSONAS, JF, JF_TO_AREA, 'persona_hike_rate', 'high_hike_rate'),
            'workstyle': _personas(D, WORKSTYLE_PERSONAS, SF, {}, 'persona_success_rate', 'success_rate'),
        },
    }


def evaluation(st):
    R = st['R']
    E = R['E']
    return {
        'n_students': E['n_students'],
        'n_simulations': E['n_sims'],
        'summary': E['summary'],
        'wins_pct': E['wins_pct'],
        'wilcoxon_p': E['wilcoxon'],
        'sensitivity': E['sensitivity'],
        'solo_hike_prob': _r(E['solo_prob_mean'], 3),
        'profile_train_test_corr': _r(R['B']['profile_train_test_corr'], 3),
    }


def transparency(st):
    R = st['R']
    A, B, C, D, E = R['A'], R['B'], R['C'], R['D'], R['E']
    log = B['cleaning_log']
    ceil_f = max(C['ceiling_pct'], key=C['ceiling_pct'].get)
    trunc = B['skills_truncated_pct']
    return {
        'cleaning_log': [{'step': s, 'rows': int(n)} for s, n in log],
        'data_issues': [
            {'issue': 'Truncated skill lists', 'evidence': f"{trunc:.1f}% of key_skills end with '...'",
             'fix': 'Kept visible skills, flagged truncation, profiles use share of postings not counts'},
            {'issue': 'Messy job titles', 'evidence': f"{B['desig_unique_raw']:,} unique designations",
             'fix': 'Rule-based mapping to 6 core data roles, skills used as fallback'},
            {'issue': 'Mostly empty job_type', 'evidence': f"{B['null_job_type_pct']:.1f}% null, {len(B['job_type_variants'])} spellings",
             'fix': 'Standardised case, column not used for modelling'},
            {'issue': 'Duplicate postings', 'evidence': f'{log[0][1] - log[1][1]:,} exact duplicates', 'fix': 'Removed'},
            {'issue': 'Non-analytics noise', 'evidence': f"{B['noise_removed']:,} data entry, BPO, telecalling postings",
             'fix': 'Keyword filter removed them'},
            {'issue': 'Salary and experience as text', 'evidence': "'7.8L', '10to15', '3-8 yrs'",
             'fix': 'Parsed to numeric min, max and midpoint'},
            {'issue': 'Extreme posting counts', 'evidence': f"num_of_jobs up to {A['num_jobs_max']:,}; {A['num_jobs_outliers']} IQR outliers",
             'fix': 'Kept (real large employers), medians used instead of means'},
            {'issue': 'Ceiling effect in JDS scores', 'evidence': f"{C['ceiling_pct'][ceil_f]:.0f}% score 5 on {CEILING_LABEL[ceil_f]}",
             'fix': 'Reported, non-parametric tests run alongside t-tests'},
        ],
        'scoring': {
            'weights': E['weights'],
            'payoff_weights': {k: _r(v, 3) for k, v in E['payoff_weights'].items()},
            'role_weights': E['role_weights_train'],
            'formula': FORMULA,
        },
        'models': [
            {'name': 'Skill payoff model', 'data': f"JDS Skill Traits ({C['n']})", 'type': 'Logistic regression',
             'cv_auc': C['models'][0]['cv_auc']},
            {'name': 'Success trait model', 'data': f"SDS Personality Traits ({D['n']})", 'type': 'Logistic regression',
             'cv_auc': D['models'][0]['cv_auc']},
        ],
        'assumptions': [
            'Student profiles are simulated from JDS and SDS distributions, shifted down to an earlier career stage',
            'Senior data scientist traits are used as indicators of success, not as a profile of students',
            'A group covers an area when its strongest member covers it (peer teaching)',
        ],
        'limitations': [
            f'{trunc:.0f}% of skill lists are truncated, so rare skills are under-counted',
            'Only 5 broad skill areas link jobs data to outcome data',
            'Simulated students, not a live user pilot',
            f"Small JDS and SDS samples ({C['n']} and {D['n']} rows)",
        ],
    }


def pool_list(st, role=None):
    p = st['pool']
    if role:
        p = p[p.role == role]
    return [{'name': r['name'], 'role': r['role'], 'city': r['city'], 'hours': int(r['hours']),
             'skills': {a: _r(r[f'sk_{i}'], 2) for i, a in enumerate(AREA_KEYS)},
             'learner_persona': r['learner_persona'], 'workstyle_persona': r['workstyle_persona'],
             'trait_success_prob': _r(r['trait_success_prob'], 3)} for _, r in p.iterrows()]


# ---------------- POST /api/match ----------------
def _city_key(c):
    c = c.strip().lower()
    return CITY_ALIAS.get(c, c)


def match(st, req):
    eng = st['engine']
    role = req['target_role']
    W = st['W_tr'][role]
    skills = {a: float(req['skills'][a]) for a in AREA_KEYS}
    user_sk = np.array([skills[a] for a in AREA_KEYS])
    traits = tipi_to_sds(req['tipi'])
    user_tr = np.array([traits[f] for f in SF], float)
    user_tp = float(eng.success_prob(user_tr)[0])
    user_learner = learner_persona(user_sk, st['jprof'], st['jnames'])
    user_style = workstyle_persona(user_tr, st['skm'], st['sksc'], st['snames'])
    gap = weighted_gap(skills, W)

    # insert the user into a copy of the pool arrays; the stored pool is never mutated
    pool = st['pool']
    n = len(pool)
    SKa = np.vstack([pool[SK].values.astype(float), user_sk])
    Ha = np.append(pool.hours.values.astype(float), float(req['hours_per_week']))
    TPa = np.append(pool.trait_success_prob.values, user_tp)
    cand = list(np.flatnonzero(pool.role.values == role))
    if len(cand) < 2:  # too few peers in this role: borrow from the whole pool
        cand = list(range(n))
    G = eng.greedy(SKa, Ha, TPa, cand + [n], W, forced_seed=n, first_only=True)[0]

    members = []
    for pos in G:
        if pos == n:
            m = {'name': req.get('name') or 'You', 'is_user': True, 'skills': {a: _r(skills[a], 2) for a in AREA_KEYS},
                 'hours_per_week': int(req['hours_per_week']), 'learner_persona': user_learner,
                 'workstyle_persona': user_style, 'trait_success_prob': _r(user_tp, 3)}
        else:
            r = pool.iloc[pos]
            m = {'name': r['name'], 'is_user': False, 'skills': {a: _r(r[f'sk_{i}'], 2) for i, a in enumerate(AREA_KEYS)},
                 'hours_per_week': int(r['hours']), 'learner_persona': r['learner_persona'],
                 'workstyle_persona': r['workstyle_persona'], 'trait_success_prob': _r(r['trait_success_prob'], 3)}
        m['strongest_area'] = max(AREA_KEYS, key=lambda a: m['skills'][a])
        members.append(m)
    lead_i = int(np.argmax(TPa[G]))
    for i, m in enumerate(members):
        m['accountability_lead'] = i == lead_i
    members = [{k: m[k] for k in ['name', 'is_user', 'skills', 'hours_per_week', 'strongest_area', 'learner_persona',
                                  'workstyle_persona', 'trait_success_prob', 'accountability_lead']} for m in members]

    parts = eng.parts(SKa[G], Ha[G], TPa[G], W)
    lead = members[lead_i]
    sentences, uncovered = explanation(members, gap, parts, lead['name'], lead['workstyle_persona'])

    city = req['city']
    ck = _city_key(city)
    city_of = {i: _city_key(c) for i, c in enumerate(pool.city.values)}
    all_groups = [g for gs in st['pool_groups'].values() for g in gs]
    groups_in_city = sum(1 for g in all_groups if sum(city_of[i] == ck for i in g) >= 2)

    return {
        'user': {
            'learner_persona': user_learner,
            'workstyle_persona': user_style,
            'traits_sds_scale': traits,
            'trait_success_prob': _r(user_tp, 3),
            'solo_hike_prob': _r(eng.hike_prob(user_sk)[0], 3),
            'gap': gap,
        },
        'group': {
            'members': members,
            'score': _r(eng.score_parts(parts), 3),
            'parts': {k: _r(v, 3) for k, v in parts.items()},
            'group_hike_prob': _r(eng.hike_prob(SKa[G].max(axis=0))[0], 3),
            'explanation': sentences,
            'uncovered_areas': uncovered,
            'accountability_note': 'Accountability lead rotates weekly',
        },
        'plan': learning_plan(members, W),
        'project': pick_project(members, W, role),
        'community': {
            'role_circle': {'role': role, 'groups': len(st['pool_groups'].get(role, [])),
                            'members': int((pool.role == role).sum())},
            'city_circle': {'city': city, 'data_role_postings': int(st['city_postings'].get(ck, 0)),
                            'groups_in_city': groups_in_city},
            'mentor_pool': MENTOR_POOL[role],
        },
    }
