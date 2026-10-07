"""Module A and B cleaning, ported from reference_analysis.py (seed 42)."""
import re

import numpy as np
import pandas as pd
from scipy import stats


# ---------------- MODULE A : DataScience_Jobs ----------------
def module_a(data_dir, out_dir):
    d = pd.read_csv(data_dir / 'DataScience_Jobs.csv')
    A = {'rows_raw': len(d), 'companies': int(d.company_name.nunique())}
    for c in ['avg_salary', 'min_salary', 'max_salary']:
        d[c + '_lpa'] = d[c].astype(str).str.replace('L', '', regex=False).astype(float)
    bad = d[(d.min_salary_lpa > d.avg_salary_lpa) | (d.avg_salary_lpa > d.max_salary_lpa)]
    A['inconsistent_salary_rows'] = len(bad)
    d['company_name'] = d.company_name.str.strip()
    A['dup_rows'] = int(d.drop(columns='reference_no').duplicated().sum())
    q1, q3 = d.num_of_jobs.quantile([.25, .75])
    A['num_jobs_outlier_threshold'] = float(q3 + 1.5 * (q3 - q1))
    A['num_jobs_outliers'] = int((d.num_of_jobs > q3 + 1.5 * (q3 - q1)).sum())
    A['num_jobs_max'] = int(d.num_of_jobs.max())
    d['base_role'] = d.job_title.str.replace('Senior ', '', regex=False)
    d['senior'] = d.job_title.str.startswith('Senior')
    role_tab = d.groupby('job_title').agg(listings=('num_of_jobs', 'sum'), companies=('company_name', 'nunique'),
                                          median_avg_salary=('avg_salary_lpa', 'median'),
                                          mean_min_exp=('min_experience', 'mean')).sort_values('listings', ascending=False)
    # attractiveness index: demand share x pay / (1+exp)
    rt = role_tab.copy()
    rt['demand_idx'] = rt.listings / rt.listings.max()
    rt['pay_idx'] = rt.median_avg_salary / rt.median_avg_salary.max()
    rt['barrier'] = rt.mean_min_exp
    rt['attractiveness'] = (rt.demand_idx * 0.5 + rt.pay_idx * 0.5) / (1 + rt.barrier / 5)
    rt['attractiveness'] = rt.attractiveness / rt.attractiveness.max() * 100
    rt = rt.round(2)
    rt.to_csv(out_dir / 'A_role_table.csv')
    A['role_table'] = rt.reset_index().to_dict('records')
    groups = [g.avg_salary_lpa.values for _, g in d.groupby('job_title')]
    A['anova_F'], A['anova_p'] = map(float, stats.f_oneway(*groups))
    A['kruskal_H'], A['kruskal_p'] = map(float, stats.kruskal(*groups))
    A['levene_p'] = float(stats.levene(*groups).pvalue)
    rho, p = stats.spearmanr(d.min_experience, d.avg_salary_lpa)
    A['spearman_exp_salary'] = [float(rho), float(p)]
    sen = d[d.senior].avg_salary_lpa
    jun = d[~d.senior].avg_salary_lpa
    A['senior_vs_junior_median'] = [float(sen.median()), float(jun.median())]
    A['mannwhitney_senior_p'] = float(stats.mannwhitneyu(sen, jun).pvalue)
    d.to_csv(out_dir / 'A_clean.csv', index=False)
    return A


# ---------------- MODULE B : Analytics_Jobs (cleaning part) ----------------
NOISE = (r'data entry|bpo|call cent|tele ?call|telecall|voice process|non voice|ar call|accounts receivable|typing|'
         r'back office|home base|part time|work from home|freelanc|ar analyst|kpo|night shift|customer care|customer support')

SYN = {
    'ml': 'machine learning', 'machine learning algorithms': 'machine learning', 'machine learning techniques': 'machine learning',
    'ai': 'artificial intelligence', 'nlp': 'natural language processing', 'dl': 'deep learning',
    'advanced excel': 'excel', 'ms excel': 'excel', 'excel macros': 'vba', 'macros': 'vba', 'ms office': 'ms office',
    'microsoft excel': 'excel', 'power bi': 'power bi', 'powerbi': 'power bi', 'power-bi': 'power bi',
    'qlikview': 'qlik', 'qlik sense': 'qlik', 'qliksense': 'qlik', 'tableau desktop': 'tableau',
    'sql server': 'sql', 'ms sql': 'sql', 'sql queries': 'sql', 'mysql': 'sql', 'pl/sql': 'plsql', 'pl sql': 'plsql', 'oracle sql': 'sql',
    'r programming': 'r', 'r language': 'r', 'python programming': 'python', 'python scripting': 'python',
    'sas programming': 'sas', 'base sas': 'sas', 'sas base': 'sas', 'sas sql': 'sas', 'advanced sas': 'sas', 'sas macros': 'sas',
    'big data analytics': 'big data', 'bigdata': 'big data', 'apache spark': 'spark', 'pyspark': 'spark', 'spark sql': 'spark',
    'hadoop ecosystem': 'hadoop', 'map reduce': 'mapreduce', 'statistical modeling': 'statistical modelling',
    'statistical analysis': 'statistics', 'statistical': 'statistics', 'data visualisation': 'data visualization',
    'visualization': 'data visualization', 'data analyst': 'data analysis', 'data analytics': 'data analysis', 'analysis': 'data analysis',
    'predictive modeling': 'predictive modelling', 'predictive analytics': 'predictive modelling', 'predictive analysis': 'predictive modelling',
    'logistic regression': 'regression', 'linear regression': 'regression', 'regression analysis': 'regression',
    'time series analysis': 'time series', 'forecasting': 'forecasting', 'data warehouse': 'data warehousing', 'dwh': 'data warehousing',
    'business intelligence': 'business intelligence', 'bi': 'business intelligence', 'amazon web services': 'aws',
    'microsoft azure': 'azure', 'google analytics': 'web analytics', 'adobe analytics': 'web analytics', 'artificial intelligence': 'artificial intelligence',
    'data scientist': 'data science', 'deep learning': 'deep learning', 'neural network': 'neural networks', 'tensor flow': 'tensorflow',
    'communication skills': 'communication', 'communication': 'communication', 'mis reporting': 'mis', 'mis reports': 'mis', 'reporting': 'reporting',
}


def norm_skill(s):
    s = s.lower().strip().strip('.').strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('&', 'and')
    return SYN.get(s, s)


ROLES = [
    ('Machine Learning Engineer', r'machine learning engineer|ml engineer|deep learning|\bnlp\b|computer vision|\bai engineer|artificial intelligence engineer|ml developer'),
    ('Data Architect', r'data architect|big data architect|solution architect.*data|data modell?er'),
    ('Data Engineer', r'data engineer|big data|hadoop|spark|\betl\b|data warehous|informatica|datastage|ab initio|pyspark|data integration|\bdwh\b'),
    ('Data Scientist', r'data scien|machine learning|predictive model|statistical model|modell?er|quant|ai\b'),
    ('Business Analyst', r'business analy|functional analy|\bba\b|business system analy'),
    ('Data Analyst', r'data analy|analytics|\bmis\b|reporting analy|\bbi\b|business intelligence|tableau|power ?bi|qlik|insight|research analy|sas|analyst'),
]


def map_role(row):
    t = str(row.job_desig).lower()
    k0 = ' '.join(row.skills)
    if re.search(r'\bseo\b|digital marketing|social media|content writ|accountant|audit|\bhr\b|recruit|sales executive|business development|tele', t):
        return 'Other'
    for r, pat in ROLES:
        if r == 'Data Analyst' and re.fullmatch(r'.*\banalyst\b.*', t) and not re.search(r'data|analytics|\bmis\b|\bbi\b|intelligence|tableau|power ?bi|qlik|insight|sas', t):
            if not re.search(r'sql|python|\br\b|sas|tableau|power bi|excel|data analysis|statistics|analytics', k0):
                continue
        if re.search(pat, t):
            return r
    # fall back on skills when title is uninformative
    k = ' '.join(row.skills)
    if re.search(r'machine learning|data science|deep learning', k) and re.search(r'python|\br\b', k):
        return 'Data Scientist'
    if re.search(r'hadoop|spark|big data|data warehousing', k):
        return 'Data Engineer'
    if re.search(r'business analysis', k):
        return 'Business Analyst'
    if re.search(r'data analysis|tableau|power bi|sql|sas|excel', k) and re.search(r'analy', t + k):
        return 'Data Analyst'
    return 'Other'


def module_b_clean(data_dir):
    """Returns (core postings, B stats dict). Cleaning log matches the reference order."""
    a = pd.read_csv(data_dir / 'Analytics_Jobs.csv')
    B = {'rows_raw': len(a)}
    log = [('Raw postings', len(a))]
    B['null_job_type_pct'] = float(a.job_type.isna().mean() * 100)
    B['null_desc_pct'] = float(a.job_description.isna().mean() * 100)
    B['job_type_variants'] = a.job_type.dropna().unique().tolist()
    B['desig_unique_raw'] = int(a.job_desig.nunique())
    B['skills_truncated_pct'] = float(a.key_skills.fillna('').str.strip().str.endswith('...').mean() * 100)
    B['desc_median_len'] = float(a.job_description.dropna().str.len().median())
    a = a.drop_duplicates(subset=[c for c in a.columns if c != 's_no'])
    log.append(('After removing exact duplicates', len(a)))
    a = a[a.key_skills.notna()].copy()
    log.append(('After dropping postings with no skills', len(a)))
    a['job_type'] = a.job_type.str.lower().str.strip().replace({'analytic': 'analytics'})
    # experience and salary
    ex = a.experience.str.extract(r'(\d+)\s*-\s*(\d+)').astype(float)
    a['exp_min'], a['exp_max'] = ex[0], ex[1]
    a['exp_mid'] = (a.exp_min + a.exp_max) / 2
    sal = a.salary.str.extract(r'(\d+)to(\d+)').astype(float)
    a['sal_min'], a['sal_max'] = sal[0], sal[1]
    a['sal_mid'] = (a.sal_min + a.sal_max) / 2
    band_order = ['0to3', '3to6', '6to10', '10to15', '15to25', '25to50']
    a['sal_band'] = pd.Categorical(a.salary, band_order, ordered=True)
    a['city'] = a.location.str.split(',').str[0].str.strip()
    a['multi_city'] = a.location.str.contains(',')
    # noise filter
    txt = (a.job_desig.fillna('') + ' | ' + a.key_skills.fillna('')).str.lower()
    a['noise'] = txt.str.contains(NOISE)
    B['noise_removed'] = int(a.noise.sum())
    a = a[~a.noise].copy()
    log.append(('After removing non-analytics noise (data entry, BPO, telecalling)', len(a)))

    # skill normalisation
    a['skills'] = a.key_skills.apply(lambda k: sorted({norm_skill(x) for x in str(k).split(',') if norm_skill(x) and len(norm_skill(x)) < 40}))
    raw_unique = len({x.strip().strip('.').lower() for k in a.key_skills for x in str(k).split(',') if x.strip().strip('.')})
    norm_unique = len({s for l in a.skills for s in l})
    B['skills_unique_raw'] = raw_unique
    B['skills_unique_norm'] = norm_unique
    a['n_skills'] = a.skills.str.len()
    B['skills_per_posting_median'] = float(a.n_skills.median())

    # role mapping
    a['role'] = a.apply(map_role, axis=1)
    a['senior'] = a.job_desig.str.lower().str.contains(r'senior|\bsr\b|lead|manager|head|principal|director|vp\b|avp')
    B['role_counts'] = a.role.value_counts().to_dict()
    core = a[a.role != 'Other'].copy()
    log.append(('Postings mapped to one of 6 core data roles', len(core)))
    B['cleaning_log'] = log
    B['desig_unique_after'] = 6
    return core, B
