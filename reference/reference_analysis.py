import pandas as pd, numpy as np, re, json, collections, itertools
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score

D = '/home/claude/data/'
OUT = '/home/claude/work/'
R = {}  # results
rng = np.random.default_rng(42)

# ---------------- MODULE A : DataScience_Jobs ----------------
d = pd.read_csv(D + 'DataScience_Jobs.csv')
A = {'rows_raw': len(d), 'companies': int(d.company_name.nunique())}
for c in ['avg_salary', 'min_salary', 'max_salary']:
    d[c + '_lpa'] = d[c].str.replace('L', '', regex=False).astype(float)
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
rt.to_csv(OUT + 'A_role_table.csv')
A['role_table'] = rt.reset_index().to_dict('records')
groups = [g.avg_salary_lpa.values for _, g in d.groupby('job_title')]
A['anova_F'], A['anova_p'] = map(float, stats.f_oneway(*groups))
A['kruskal_H'], A['kruskal_p'] = map(float, stats.kruskal(*groups))
A['levene_p'] = float(stats.levene(*groups).pvalue)
rho, p = stats.spearmanr(d.min_experience, d.avg_salary_lpa)
A['spearman_exp_salary'] = [float(rho), float(p)]
sen = d[d.senior].avg_salary_lpa; jun = d[~d.senior].avg_salary_lpa
A['senior_vs_junior_median'] = [float(sen.median()), float(jun.median())]
A['mannwhitney_senior_p'] = float(stats.mannwhitneyu(sen, jun).pvalue)
d.to_csv(OUT + 'A_clean.csv', index=False)
R['A'] = A

# ---------------- MODULE B : Analytics_Jobs ----------------
a = pd.read_csv(D + 'Analytics_Jobs.csv')
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
a = a[a.key_skills.notna()]
log.append(('After dropping postings with no skills', len(a)))
a['job_type'] = a.job_type.str.lower().str.strip().replace({'analytic': 'analytics'})
# experience & salary
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
noise = r'data entry|bpo|call cent|tele ?call|telecall|voice process|non voice|ar call|accounts receivable|typing|back office|home base|part time|work from home|freelanc|ar analyst|kpo|night shift|customer care|customer support'
txt = (a.job_desig.fillna('') + ' | ' + a.key_skills.fillna('')).str.lower()
a['noise'] = txt.str.contains(noise)
B['noise_removed'] = int(a.noise.sum())
a = a[~a.noise].copy()
log.append(('After removing non-analytics noise (data entry, BPO, telecalling)', len(a)))

# skill normalisation
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
a['skills'] = a.key_skills.apply(lambda k: sorted({norm_skill(x) for x in str(k).split(',') if norm_skill(x) and len(norm_skill(x)) < 40}))
raw_unique = len({x.strip().strip('.').lower() for k in a.key_skills for x in str(k).split(',') if x.strip().strip('.')})
norm_unique = len({s for l in a.skills for s in l})
B['skills_unique_raw'] = raw_unique; B['skills_unique_norm'] = norm_unique
a['n_skills'] = a.skills.str.len()
B['skills_per_posting_median'] = float(a.n_skills.median())

# role mapping
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
a['role'] = a.apply(map_role, axis=1)
a['senior'] = a.job_desig.str.lower().str.contains(r'senior|\bsr\b|lead|manager|head|principal|director|vp\b|avp')
B['role_counts'] = a.role.value_counts().to_dict()
core = a[a.role != 'Other'].copy()
log.append(('Postings mapped to one of 6 core data roles', len(core)))
B['cleaning_log'] = log
B['desig_unique_after'] = 6

# 5 skill areas (bridge to JDS)
AREAS = {
    'big_data': r'hadoop|spark|big data|hive|\bpig\b|scala|kafka|nosql|cassandra|mongodb|\betl\b|data warehousing|aws|azure|cloud|informatica|mapreduce|hbase|data engineering|gcp|snowflake|databricks|teradata|data lake',
    'maths_stats': r'statistic|regression|anova|hypothesis|econometric|forecasting|time series|quantitative|mathemat|probabilit|spss|chaid|segmentation|clustering|optimization|operations research|stata|eviews|multivariate|sampling',
    'coding': r'^python$|^r$|^sql$|^sas$|^java$|^c$|c\+\+|^vba$|matlab|perl|shell script|plsql|programming|^scala$|coding|^unix$|linux',
    'ai_ml': r'machine learning|deep learning|artificial intelligence|natural language|computer vision|neural|tensorflow|keras|pytorch|predictive modelling|data science|text mining|scikit|random forest|xgboost|data mining',
    'dashboard_story': r'tableau|power bi|qlik|data visualization|dashboard|reporting|^mis$|^excel$|business intelligence|storytelling|presentation|spotfire|cognos|microstrategy|powerpoint|ppt|ssrs|crystal reports|visual analytics|insight',
}
AREA_LABEL = {'big_data': 'Big data', 'maths_stats': 'Maths and stats', 'coding': 'Coding', 'ai_ml': 'AI and ML', 'dashboard_story': 'Dashboards and storytelling'}
def areas_of(skills):
    hit = set()
    for s in skills:
        for ar, pat in AREAS.items():
            if re.search(pat, s):
                hit.add(ar)
    return hit
core['areas'] = core.skills.apply(areas_of)
for ar in AREAS:
    core[ar] = core.areas.apply(lambda s: ar in s).astype(int)
mapped_skill_count = collections.Counter()
for l in core.skills:
    for s in l:
        mapped_skill_count[(s, bool(areas_of([s])))] += 1
tot = sum(mapped_skill_count.values()); hit = sum(v for (s, m), v in mapped_skill_count.items() if m)
B['skill_mentions_mapped_to_area_pct'] = hit / tot * 100
B['postings_with_any_area_pct'] = float((core[list(AREAS)].sum(axis=1) > 0).mean() * 100)

# train / test split of postings
core['split'] = np.where(rng.random(len(core)) < 0.8, 'train', 'test')
tr = core[core.split == 'train']; te = core[core.split == 'test']
B['train_n'], B['test_n'] = len(tr), len(te)

def area_profile(df):
    return df.groupby('role')[list(AREAS)].mean()
prof_tr = area_profile(tr); prof_te = area_profile(te)
prof_all = area_profile(core)
prof_all.round(3).to_csv(OUT + 'B_area_profile.csv')
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
            share = k / n; base = all_counts[s] / N
            lift_rows.append({'role': role, 'skill': s, 'postings': k, 'share_pct': share * 100, 'lift': share / base})
lift = pd.DataFrame(lift_rows)
lift['score'] = lift.share_pct / 100 * np.log1p(lift.lift)  # demand weighted distinctiveness
top_lift = lift.sort_values(['role', 'score'], ascending=[True, False]).groupby('role').head(8)
top_lift.round(2).to_csv(OUT + 'B_top_skills.csv', index=False)
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
    if sup < 0.01: continue
    for ant, con in [(x, y), (y, x)]:
        conf = c / item_c[ant]; lft = conf / (item_c[con] / N)
        rules.append({'if': ant, 'then': con, 'support_pct': sup * 100, 'confidence_pct': conf * 100, 'lift': lft})
rules = pd.DataFrame(rules).sort_values('lift', ascending=False)
rules = rules[rules.confidence_pct >= 30]
rules.head(12).round(2).to_csv(OUT + 'B_rules.csv', index=False)
B['rules'] = rules.head(10).round(2).to_dict('records')
B['n_rules'] = len(rules)

# salary vs role (Kruskal) and areas vs salary band (chi-square)
B['role_salary_median'] = core.groupby('role').sal_mid.median().to_dict()
B['kruskal_role_salary'] = list(map(float, stats.kruskal(*[g.sal_mid.dropna() for _, g in core.groupby('role')])))
chi = {}
for ar in AREAS:
    ct = pd.crosstab(core[ar], core.sal_band)
    c2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum(); v = np.sqrt(c2 / (n * (min(ct.shape) - 1)))
    hi = core[core[ar] == 1].sal_mid.median(); lo = core[core[ar] == 0].sal_mid.median()
    share_high_with = (core[core[ar] == 1].sal_min >= 15).mean() * 100
    share_high_without = (core[core[ar] == 0].sal_min >= 15).mean() * 100
    chi[ar] = {'chi2': float(c2), 'p': float(p), 'cramers_v': float(v), 'high_band_with': float(share_high_with), 'high_band_without': float(share_high_without)}
B['chi_area_salary'] = chi
rho, p = stats.spearmanr(core.exp_mid, core.sal_mid, nan_policy='omit')
B['spearman_exp_salary'] = [float(rho), float(p)]
B['top_cities'] = core.city.value_counts().head(8).to_dict()
B['city_role'] = pd.crosstab(core.city, core.role).loc[list(core.city.value_counts().head(6).index)].to_dict('index')
B['seniority_share_pct'] = float(core.senior.mean() * 100)
B['median_exp_by_role'] = core.groupby('role').exp_min.median().to_dict()
core.drop(columns=['areas']).assign(skills=core.skills.str.join('; ')).to_csv(OUT + 'B_clean_core.csv', index=False)
R['B'] = B

# ---------------- MODULE C / D : JDS & SDS ----------------
def wald_logit(X, y):
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(Xs, y)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    p = m.predict_proba(Xs)[:, 1]
    W = p * (1 - p)
    cov = np.linalg.inv(Xd.T @ (Xd * W[:, None]))
    beta = np.concatenate([m.intercept_, m.coef_[0]])
    se = np.sqrt(np.diag(cov)); z = beta / se; pv = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)); p0 = y.mean()
    ll0 = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    return m, sc, beta, se, pv, 1 - ll / ll0

def classify_module(df, feats, target, label):
    X = df[feats].values; y = df[target].values
    M = {'n': len(df), 'class_balance': df[target].value_counts().to_dict()}
    tt = []
    for f in feats:
        h = df[df[target] == 1][f]; l = df[df[target] == 0][f]
        t, p = stats.ttest_ind(h, l, equal_var=False)
        u = stats.mannwhitneyu(h, l).pvalue
        dpool = (h.mean() - l.mean()) / np.sqrt((h.var() + l.var()) / 2)
        tt.append({'feature': f, 'mean_high': h.mean(), 'mean_low': l.mean(), 't': t, 'p': p, 'mw_p': u, 'cohens_d': dpool})
    M['ttests'] = pd.DataFrame(tt).round(4).to_dict('records')
    m, sc, beta, se, pv, r2 = wald_logit(X, y)
    M['logit'] = [{'feature': f, 'coef_std': float(b), 'odds_ratio_per_sd': float(np.exp(b)), 'p': float(p)} for f, b, p in zip(['intercept'] + feats, beta, pv)]
    M['mcfadden_r2'] = float(r2)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    models = {
        'Logistic regression': (LogisticRegression(max_iter=5000), True),
        'Decision tree (depth 3)': (DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42), False),
        'Random forest': (RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=42), False),
    }
    perf = []
    for name, (mdl, scale) in models.items():
        Xm = StandardScaler().fit_transform(X) if scale else X
        prob = cross_val_predict(mdl, Xm, y, cv=cv, method='predict_proba')[:, 1]
        perf.append({'model': name, 'cv_auc': roc_auc_score(y, prob), 'cv_accuracy': accuracy_score(y, prob >= .5)})
    M['models'] = pd.DataFrame(perf).round(3).to_dict('records')
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=42).fit(X, y)
    M['rf_importance'] = dict(zip(feats, rf.feature_importances_.round(4)))
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42).fit(X, y)
    from sklearn.tree import export_text
    M['tree_text'] = export_text(dt, feature_names=feats)
    M['corr'] = df[feats + [target]].corr().round(3).to_dict()
    # personas
    Xs = StandardScaler().fit_transform(X)
    sil = {}
    for k in range(2, 7):
        lab = KMeans(k, n_init=20, random_state=42).fit_predict(Xs)
        sil[k] = silhouette_score(Xs, lab)
    inertia = {k: float(KMeans(k, n_init=20, random_state=42).fit(Xs).inertia_) for k in range(1, 8)}
    M['silhouette'] = sil; M['inertia'] = inertia
    return M, m, sc

j = pd.read_excel(D + 'JDS_Skill_Traits.xlsx')
j.columns = [c.strip().replace('-', '_') for c in j.columns]
JF = ['big_data_skills', 'maths_stats_skills', 'coding_skills', 'ai_and_ml_skills', 'dashboard_and_storytelling_skills']
C, jm, jsc = classify_module(j, JF, 'salary_hike_high_or_low', 'JDS')
C['dup_ids'] = int(j.id.duplicated().sum()); C['out_of_range'] = int(((j[JF] < 1) | (j[JF] > 5)).sum().sum())
C['ceiling_pct'] = {f: float((j[f] == 5).mean() * 100) for f in JF}
C['describe'] = j[JF].describe().round(2).to_dict()
s = pd.read_excel(D + 'SDS_Personality_Traits.xlsx')
s.columns = [c.strip().replace(' ', '') for c in s.columns]
s = s.rename(columns={'success_classification_high_low': 'success'})
SF = ['neuroticism', 'extraversion', 'openness_to_experience', 'agreeableness', 'conscientiousness']
Dm, sm, ssc = classify_module(s, SF, 'success', 'SDS')
Dm['dup_ids'] = int(s.id.duplicated().sum())
Dm['describe'] = s[SF].describe().round(2).to_dict()
Dm['iqr_outliers'] = {f: int(((s[f] < s[f].quantile(.25) - 1.5 * (s[f].quantile(.75) - s[f].quantile(.25))) | (s[f] > s[f].quantile(.75) + 1.5 * (s[f].quantile(.75) - s[f].quantile(.25)))).sum()) for f in SF}

# personas: pick k for interpretability within best silhouette
def personas(df, feats, k, names_fn):
    sc = StandardScaler().fit(df[feats]); km = KMeans(k, n_init=20, random_state=42).fit(sc.transform(df[feats]))
    df = df.copy(); df['cluster'] = km.labels_
    prof = df.groupby('cluster')[feats].mean()
    return df, prof, km, sc
kj = max(C['silhouette'], key=C['silhouette'].get); kj = kj if kj >= 3 else 3
jd, jprof, jkm, jksc = personas(j, JF, kj, None)
ks = max(Dm['silhouette'], key=Dm['silhouette'].get); ks = ks if ks >= 3 else 3
sd, sprof, skm, sksc = personas(s, SF, ks, None)
C['k'] = kj; Dm['k'] = ks
C['persona_profile'] = jprof.round(2).to_dict('index')
C['persona_hike_rate'] = jd.groupby('cluster').salary_hike_high_or_low.mean().round(3).to_dict()
C['persona_size'] = jd.cluster.value_counts().to_dict()
Dm['persona_profile'] = sprof.round(2).to_dict('index')
Dm['persona_success_rate'] = sd.groupby('cluster').success.mean().round(3).to_dict()
Dm['persona_size'] = sd.cluster.value_counts().to_dict()
R['C'] = C; R['D'] = Dm

# ---------------- MODULE E : matching engine ----------------
# payoff weights from JDS logistic (std coefficients, floor at small positive)
coef = dict(zip(JF, jm.coef_[0]))
area_from_jds = dict(zip(JF, AREAS))
payoff = {area_from_jds[f]: max(coef[f], 0) for f in JF}
mx = max(payoff.values()) or 1
payoff = {k: 0.5 + 0.5 * v / mx for k, v in payoff.items()}  # 0.5..1
E = {'payoff_weights': payoff}

def role_weights(prof):
    W = {}
    for role, row in prof.iterrows():
        w = np.array([row[ar] * payoff[ar] for ar in AREAS]) + 1e-6
        W[role] = w / w.sum()
    return W
W_tr = role_weights(prof_tr); W_te = role_weights(prof_te)
E['role_weights_train'] = {r: dict(zip(AREAS, np.round(w, 3))) for r, w in W_tr.items()}

# role demand for simulated targets (postings share)
role_share = core.role.value_counts(normalize=True)
E['role_share'] = role_share.round(3).to_dict()

def simulate_students(n, seed):
    g = np.random.default_rng(seed)
    js = j[JF].values[g.integers(0, len(j), n)]
    # students are earlier in their journey: shift down by U(0.5,1.5), jitter, clip to 1..5
    skills = np.clip(js - g.uniform(0.5, 1.5, (n, 1)) + g.normal(0, 0.4, js.shape), 1, 5)
    tr_ = s[SF].values[g.integers(0, len(s), n)] + g.normal(0, 2, (n, 5))
    hours = g.choice([4, 6, 8, 10, 12, 15], n, p=[.15, .2, .25, .2, .12, .08])
    roles = g.choice(role_share.index, n, p=role_share.values)
    return pd.DataFrame({'role': roles, 'hours': hours, **{f'sk_{i}': skills[:, i] for i in range(5)}, **{f'tr_{i}': tr_[:, i] for i in range(5)}})

SK = [f'sk_{i}' for i in range(5)]; TR = [f'tr_{i}' for i in range(5)]
def success_prob(tr_vals):
    return sm.predict_proba(ssc.transform(tr_vals))[:, 1]

WEIGHTS = {'coverage': 0.45, 'overlap': 0.20, 'hours': 0.15, 'traits': 0.20}
def group_parts(G, W):
    sk = G[SK].values; covered = sk.max(axis=0)
    coverage = float(np.dot(W, covered / 5))
    if len(G) > 1:
        c = sk - sk.mean(axis=1, keepdims=True)
        sims = [np.dot(c[i], c[k]) / (np.linalg.norm(c[i]) * np.linalg.norm(c[k]) + 1e-9) for i, k in itertools.combinations(range(len(G)), 2)]
        overlap = (np.mean(sims) + 1) / 2
    else:
        overlap = 1.0
    h = G.hours.values; hours = 1 - (h.max() - h.min()) / 11
    tp = success_prob(G[TR].values)
    traits = float(0.7 * tp.max() + 0.3 * tp.mean())
    return {'coverage': coverage, 'overlap': overlap, 'hours': hours, 'traits': traits}
def group_score(G, W):
    p = group_parts(G, W)
    return WEIGHTS['coverage'] * p['coverage'] + WEIGHTS['overlap'] * (1 - p['overlap']) + WEIGHTS['hours'] * p['hours'] + WEIGHTS['traits'] * p['traits']

def sizes(n):
    if n < 2: return [n]
    k = n // 3; r = n % 3
    if r == 0: return [3] * k
    if r == 1: return [3] * (k - 1) + [4]
    return [3] * k + [2]

def greedy_match(df, W):
    pool = list(df.index); groups = []
    W_ = W
    gap = ((5 - df[SK].values) / 4 * W_).sum(axis=1)
    order = list(df.index[np.argsort(-gap)])
    for sz in sizes(len(pool)):
        seed = [i for i in order if i in pool][0]; G = [seed]; pool.remove(seed)
        while len(G) < sz:
            best = max(pool, key=lambda c: group_score(df.loc[G + [c]], W_))
            G.append(best); pool.remove(best)
        groups.append(G)
    return groups
def random_match(df, seed):
    idx = list(np.random.default_rng(seed).permutation(df.index)); out = []; p = 0
    for sz in sizes(len(idx)):
        out.append(idx[p:p + sz]); p += sz
    return out
def similarity_match(df):
    pool = list(df.index); out = []
    for sz in sizes(len(pool)):
        seed = pool[0]; pool.remove(seed); G = [seed]
        while len(G) < sz:
            v = df.loc[seed, SK].values.astype(float)
            best = min(pool, key=lambda c: np.linalg.norm(df.loc[c, SK].values.astype(float) - v))
            G.append(best); pool.remove(best)
        out.append(G)
    return out

def jds_group_prob(G):
    # combined group skill (best member per area) fed to the JDS model
    v = G[SK].values.max(axis=0)[None, :]
    return float(jm.predict_proba(jsc.transform(v))[0, 1])
def solo_prob(df):
    return jm.predict_proba(jsc.transform(df[SK].values))[:, 1]

def evaluate(groups_by_role, students, W):
    rows = []
    for role, groups in groups_by_role.items():
        for G in groups:
            g = students.loc[G]; p = group_parts(g, W[role])
            rows.append({**p, 'anchor': float(success_prob(g[TR].values).max() >= 0.5), 'score': group_score(g, W[role]), 'jds_prob': jds_group_prob(g), 'size': len(G)})
    return pd.DataFrame(rows)

sims = []
N_STUD, N_SIM = 150, 30
for sim in range(N_SIM):
    st = simulate_students(N_STUD, 1000 + sim)
    res = {}
    for method in ['Our matcher', 'Random grouping', 'Similarity-only']:
        gb = {}
        for role, g in st.groupby('role'):
            if len(g) < 2: continue
            if method == 'Our matcher': gb[role] = greedy_match(g, W_tr[role])
            elif method == 'Random grouping': gb[role] = random_match(g, sim)
            else: gb[role] = similarity_match(g)
        ev = evaluate(gb, st, W_te)  # judged on held-out posting weights
        res[method] = ev.mean(numeric_only=True)
    solo = solo_prob(st).mean()
    for m_, v in res.items():
        sims.append({'sim': sim, 'method': m_, **v.to_dict(), 'solo_prob': solo})
sims = pd.DataFrame(sims)
sims.to_csv(OUT + 'E_sims.csv', index=False)
summ = sims.groupby('method')[['coverage', 'overlap', 'hours', 'traits', 'anchor', 'score', 'jds_prob']].agg(['mean', 'std']).round(4)
E['summary'] = {m_: {k: [float(summ.loc[m_, (k, 'mean')]), float(summ.loc[m_, (k, 'std')])] for k in ['coverage', 'overlap', 'hours', 'traits', 'anchor', 'score', 'jds_prob']} for m_ in summ.index}
E['solo_prob_mean'] = float(sims.solo_prob.mean())
piv = sims.pivot(index='sim', columns='method')
tests = {}
for base in ['Random grouping', 'Similarity-only']:
    for k in ['score', 'coverage', 'jds_prob', 'overlap', 'hours', 'anchor']:
        tests[f'{base}|{k}'] = float(stats.wilcoxon(piv[(k, 'Our matcher')], piv[(k, base)]).pvalue)
E['wilcoxon'] = tests
E['wins_pct'] = {base: float((piv[('score', 'Our matcher')] > piv[('score', base)]).mean() * 100) for base in ['Random grouping', 'Similarity-only']}
E['n_students'] = N_STUD; E['n_sims'] = N_SIM; E['weights'] = dict(WEIGHTS)

# sensitivity: change weights +/-
sens = []
st = simulate_students(N_STUD, 777)
for name, wts in {'Base': dict(WEIGHTS), 'Coverage heavy': {'coverage': .6, 'overlap': .15, 'hours': .1, 'traits': .15},
                  'Equal weights': {'coverage': .25, 'overlap': .25, 'hours': .25, 'traits': .25},
                  'No traits': {'coverage': .55, 'overlap': .25, 'hours': .2, 'traits': 0}}.items():
    WEIGHTS.clear(); WEIGHTS.update(wts)
    gb = {role: greedy_match(g, W_tr[role]) for role, g in st.groupby('role') if len(g) >= 2}
    ev = evaluate(gb, st, W_te)
    sens.append({'setting': name, 'coverage': ev.coverage.mean(), 'jds_prob': ev.jds_prob.mean(), 'traits': ev.traits.mean(), 'anchor': ev.anchor.mean(), 'hours': ev.hours.mean(), 'overlap': ev.overlap.mean()})
WEIGHTS.clear(); WEIGHTS.update({'coverage': 0.45, 'overlap': 0.20, 'hours': 0.15, 'traits': 0.20})
E['sensitivity'] = pd.DataFrame(sens).round(4).to_dict('records')

# worked example: one group for Data Scientist
st = simulate_students(N_STUD, 2026)
g = st[st.role == 'Data Scientist']
groups = greedy_match(g, W_tr['Data Scientist'])
G = st.loc[groups[0]]
E['example'] = {'members': G[SK + TR + ['hours']].round(2).to_dict('records'), 'parts': group_parts(G, W_tr['Data Scientist']),
                'score': group_score(G, W_tr['Data Scientist']), 'jds_prob': jds_group_prob(G),
                'solo_probs': solo_prob(G).round(3).tolist(), 'trait_probs': success_prob(G[TR].values).round(3).tolist(),
                'weights': dict(zip(AREAS, np.round(W_tr['Data Scientist'], 3)))}
R['E'] = E
R['AREA_LABEL'] = AREA_LABEL

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(x) for x in o]
    return o
json.dump(conv(R), open(OUT + 'results.json', 'w'), indent=1, default=str)
# save cleaned for charts
jd.to_csv(OUT + 'C_jds_personas.csv', index=False); sd.to_csv(OUT + 'D_sds_personas.csv', index=False)
print('done')
