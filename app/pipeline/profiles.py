"""Role area profiles, lift, association rules, chi-square (Module B analysis part)."""
import collections
import itertools
import re

import numpy as np
import pandas as pd
from scipy import stats

AREAS = {
    'big_data': r'hadoop|spark|big data|hive|\bpig\b|scala|kafka|nosql|cassandra|mongodb|\betl\b|data warehousing|aws|azure|cloud|informatica|mapreduce|hbase|data engineering|gcp|snowflake|databricks|teradata|data lake',
    'maths_stats': r'statistic|regression|anova|hypothesis|econometric|forecasting|time series|quantitative|mathemat|probabilit|spss|chaid|segmentation|clustering|optimization|operations research|stata|eviews|multivariate|sampling',
    'coding': r'^python$|^r$|^sql$|^sas$|^java$|^c$|c\+\+|^vba$|matlab|perl|shell script|plsql|programming|^scala$|coding|^unix$|linux',
    'ai_ml': r'machine learning|deep learning|artificial intelligence|natural language|computer vision|neural|tensorflow|keras|pytorch|predictive modelling|data science|text mining|scikit|random forest|xgboost|data mining',
    'dashboard_story': r'tableau|power bi|qlik|data visualization|dashboard|reporting|^mis$|^excel$|business intelligence|storytelling|presentation|spotfire|cognos|microstrategy|powerpoint|ppt|ssrs|crystal reports|visual analytics|insight',
}
AREA_KEYS = list(AREAS)
AREA_LABEL = {'big_data': 'Big data', 'maths_stats': 'Maths and stats', 'coding': 'Coding', 'ai_ml': 'AI and ML',
              'dashboard_story': 'Dashboards and storytelling'}


def areas_of(skills):
    hit = set()
    for s in skills:
        for ar, pat in AREAS.items():
            if re.search(pat, s):
                hit.add(ar)
    return hit


def area_profile(df):
    return df.groupby('role')[AREA_KEYS].mean()


def module_b_analysis(core, B, rng, out_dir):
    """Mutates core (adds area columns and split) and B. Returns (core, prof_tr, prof_te, prof_all)."""
    core['areas'] = core.skills.apply(areas_of)
    for ar in AREAS:
        core[ar] = core.areas.apply(lambda s: ar in s).astype(int)
    mapped_skill_count = collections.Counter()
    for l in core.skills:
        for s in l:
            mapped_skill_count[(s, bool(areas_of([s])))] += 1
    tot = sum(mapped_skill_count.values())
    hit = sum(v for (s, m), v in mapped_skill_count.items() if m)
    B['skill_mentions_mapped_to_area_pct'] = hit / tot * 100
    B['postings_with_any_area_pct'] = float((core[AREA_KEYS].sum(axis=1) > 0).mean() * 100)

    # train / test split of postings
    core['split'] = np.where(rng.random(len(core)) < 0.8, 'train', 'test')
    tr = core[core.split == 'train']
    te = core[core.split == 'test']
    B['train_n'], B['test_n'] = len(tr), len(te)

    prof_tr = area_profile(tr)
    prof_te = area_profile(te)
    prof_all = area_profile(core)
    prof_all.round(3).to_csv(out_dir / 'B_area_profile.csv')
    B['area_profile'] = prof_all.round(3).to_dict('index')
    B['profile_train_test_corr'] = float(np.corrcoef(prof_tr.values.ravel(), prof_te.reindex(prof_tr.index).values.ravel())[0, 1])

    # skill lift per role
    all_counts = collections.Counter(s for l in core.skills for s in l)
    N = len(core)
    lift_rows = []
    for role, g in core.groupby('role'):
        c = collections.Counter(s for l in g.skills for s in l)
        n = len(g)
        for s, k in c.items():
            if k >= max(5, 0.03 * n):
                share = k / n
                base = all_counts[s] / N
                lift_rows.append({'role': role, 'skill': s, 'postings': k, 'share_pct': share * 100, 'lift': share / base})
    lift = pd.DataFrame(lift_rows)
    lift['score'] = lift.share_pct / 100 * np.log1p(lift.lift)  # demand weighted distinctiveness
    top_lift = lift.sort_values(['role', 'score'], ascending=[True, False]).groupby('role').head(8)
    top_lift.round(2).to_csv(out_dir / 'B_top_skills.csv', index=False)
    B['top_skills'] = {r: g[['skill', 'share_pct', 'lift']].round(2).to_dict('records') for r, g in top_lift.groupby('role')}
    B['top_skills_overall'] = [[s, c] for s, c in all_counts.most_common(15)]

    # association rules (pairs) across core postings
    sets = core.skills.tolist()
    item_c = collections.Counter(s for l in sets for s in l)
    freq = {s for s, c in item_c.items() if c / N >= 0.02}
    pair_c = collections.Counter()
    for l in sets:
        f = [s for s in l if s in freq]
        for x, y in itertools.combinations(sorted(f), 2):
            pair_c[(x, y)] += 1
    rules = []
    for (x, y), c in pair_c.items():
        sup = c / N
        if sup < 0.01:
            continue
        for ant, con in [(x, y), (y, x)]:
            conf = c / item_c[ant]
            lft = conf / (item_c[con] / N)
            rules.append({'if': ant, 'then': con, 'support_pct': sup * 100, 'confidence_pct': conf * 100, 'lift': lft})
    rules = pd.DataFrame(rules).sort_values('lift', ascending=False)
    rules = rules[rules.confidence_pct >= 30]
    rules.head(12).round(2).to_csv(out_dir / 'B_rules.csv', index=False)
    B['rules'] = rules.head(10).round(2).to_dict('records')
    B['n_rules'] = len(rules)

    # salary vs role (Kruskal) and areas vs salary band (chi-square)
    B['role_salary_median'] = core.groupby('role').sal_mid.median().to_dict()
    B['kruskal_role_salary'] = list(map(float, stats.kruskal(*[g.sal_mid.dropna() for _, g in core.groupby('role')])))
    chi = {}
    for ar in AREAS:
        ct = pd.crosstab(core[ar], core.sal_band)
        ct = ct.loc[:, ct.sum(axis=0) > 0]
        c2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = np.sqrt(c2 / (n * (min(ct.shape) - 1)))
        share_high_with = (core[core[ar] == 1].sal_min >= 15).mean() * 100
        share_high_without = (core[core[ar] == 0].sal_min >= 15).mean() * 100
        chi[ar] = {'chi2': float(c2), 'p': float(p), 'cramers_v': float(v),
                   'high_band_with': float(share_high_with), 'high_band_without': float(share_high_without)}
    B['chi_area_salary'] = chi
    rho, p = stats.spearmanr(core.exp_mid, core.sal_mid, nan_policy='omit')
    B['spearman_exp_salary'] = [float(rho), float(p)]
    B['top_cities'] = core.city.value_counts().head(8).to_dict()
    B['city_role'] = pd.crosstab(core.city, core.role).loc[list(core.city.value_counts().head(6).index)].to_dict('index')
    B['seniority_share_pct'] = float(core.senior.mean() * 100)
    B['median_exp_by_role'] = core.groupby('role').exp_min.median().to_dict()
    core.drop(columns=['areas']).assign(skills=core.skills.str.join('; ')).to_csv(out_dir / 'B_clean_core.csv', index=False)
    return core, prof_tr, prof_te, prof_all
