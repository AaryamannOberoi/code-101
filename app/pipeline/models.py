"""Module C (JDS) and D (SDS): tests, Wald logistic, CV models, k-means personas."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

JF = ['big_data_skills', 'maths_stats_skills', 'coding_skills', 'ai_and_ml_skills', 'dashboard_and_storytelling_skills']
SF = ['neuroticism', 'extraversion', 'openness_to_experience', 'agreeableness', 'conscientiousness']

# Persona names, keyed by the seed-42 centroids. Clusters are matched to these by nearest centroid,
# so the names stay right even if k-means ids shift between library versions.
LEARNER_PERSONAS = [
    ('Foundation Builder', 'Even, mid-level skills across the board; AI and ML is the biggest gap.', [3.81, 3.64, 3.97, 3.23, 3.69]),
    ('All-Rounder', 'Strong in maths, coding, AI and storytelling; big data is the only soft spot.', [3.83, 4.72, 4.82, 4.83, 4.88]),
    ('AI Explorer', 'Strong AI and ML interest, lighter on coding and stats foundations.', [3.91, 3.81, 3.33, 4.84, 3.69]),
]
WORKSTYLE_PERSONAS = [
    ('Steady Supporter', 'Warm and cooperative, prefers a clear structure set by others.', [35.04, 38.17, 27.91, 56.74, 33.26]),
    ('Driver', 'Organised, open and outgoing; the natural accountability lead.', [33.7, 49.67, 47.94, 48.32, 53.72]),
    ('Quiet Reserve', 'Reserved and low-key; does best with a fixed weekly rhythm.', [30.75, 32.0, 29.96, 30.18, 34.86]),
    ('Restless Explorer', 'Curious but restless; benefits from a steady partner.', [54.27, 36.86, 43.36, 35.41, 36.86]),
]


def wald_logit(X, y):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(Xs, y)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    p = m.predict_proba(Xs)[:, 1]
    W = p * (1 - p)
    cov = np.linalg.inv(Xd.T @ (Xd * W[:, None]))
    beta = np.concatenate([m.intercept_, m.coef_[0]])
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    p0 = y.mean()
    ll0 = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    return m, sc, beta, se, pv, 1 - ll / ll0


def classify_module(df, feats, target):
    X = df[feats].values
    y = df[target].values
    M = {'n': len(df), 'class_balance': df[target].value_counts().to_dict()}
    tt = []
    for f in feats:
        h = df[df[target] == 1][f]
        l = df[df[target] == 0][f]
        t, p = stats.ttest_ind(h, l, equal_var=False)
        u = stats.mannwhitneyu(h, l).pvalue
        dpool = (h.mean() - l.mean()) / np.sqrt((h.var() + l.var()) / 2)
        tt.append({'feature': f, 'mean_high': h.mean(), 'mean_low': l.mean(), 't': t, 'p': p, 'mw_p': u, 'cohens_d': dpool})
    M['ttests'] = pd.DataFrame(tt).round(4).to_dict('records')
    m, sc, beta, se, pv, r2 = wald_logit(X, y)
    M['logit'] = [{'feature': f, 'coef_std': float(b), 'odds_ratio_per_sd': float(np.exp(b)), 'p': float(p)}
                  for f, b, p in zip(['intercept'] + feats, beta, pv)]
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
    M['tree_text'] = export_text(dt, feature_names=feats)
    M['corr'] = df[feats + [target]].corr().round(3).to_dict()
    # personas
    Xs = StandardScaler().fit_transform(X)
    sil = {}
    for k in range(2, 7):
        lab = KMeans(k, n_init=20, random_state=42).fit_predict(Xs)
        sil[k] = silhouette_score(Xs, lab)
    M['silhouette'] = sil
    M['inertia'] = {k: float(KMeans(k, n_init=20, random_state=42).fit(Xs).inertia_) for k in range(1, 8)}
    return M, m, sc


def personas(df, feats, k):
    sc = StandardScaler().fit(df[feats].values)
    km = KMeans(k, n_init=20, random_state=42).fit(sc.transform(df[feats].values))
    df = df.copy()
    df['cluster'] = km.labels_
    prof = df.groupby('cluster')[feats].mean()
    return df, prof, km, sc


def name_clusters(prof, catalogue):
    """Map cluster id -> catalogue index by nearest centroid (one to one)."""
    ref = np.array([c[2] for c in catalogue])
    cost = np.linalg.norm(prof.values[:, None, :] - ref[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return {int(prof.index[r]): int(c) for r, c in zip(rows, cols)}


def modules_cd(data_dir):
    j = pd.read_excel(data_dir / 'JDS_Skill_Traits.xlsx')
    j.columns = [c.strip().replace('-', '_') for c in j.columns]
    C, jm, jsc = classify_module(j, JF, 'salary_hike_high_or_low')
    C['dup_ids'] = int(j.id.duplicated().sum())
    C['out_of_range'] = int(((j[JF] < 1) | (j[JF] > 5)).sum().sum())
    C['ceiling_pct'] = {f: float((j[f] == 5).mean() * 100) for f in JF}
    C['describe'] = j[JF].describe().round(2).to_dict()

    s = pd.read_excel(data_dir / 'SDS_Personality_Traits.xlsx')
    s.columns = [c.strip().replace(' ', '') for c in s.columns]
    s = s.rename(columns={'success_classification_high_low': 'success'})
    Dm, sm, ssc = classify_module(s, SF, 'success')
    Dm['dup_ids'] = int(s.id.duplicated().sum())
    Dm['describe'] = s[SF].describe().round(2).to_dict()
    Dm['iqr_outliers'] = {f: int(((s[f] < s[f].quantile(.25) - 1.5 * (s[f].quantile(.75) - s[f].quantile(.25))) |
                                   (s[f] > s[f].quantile(.75) + 1.5 * (s[f].quantile(.75) - s[f].quantile(.25)))).sum()) for f in SF}

    # fixed k as validated in the reference run: JDS k=3, SDS k=4
    kj, ks = 3, 4
    jd, jprof, jkm, jksc = personas(j, JF, kj)
    sd, sprof, skm, sksc = personas(s, SF, ks)
    C['k'] = kj
    Dm['k'] = ks
    C['persona_profile'] = jprof.round(2).to_dict('index')
    C['persona_hike_rate'] = jd.groupby('cluster').salary_hike_high_or_low.mean().round(3).to_dict()
    C['persona_size'] = jd.cluster.value_counts().to_dict()
    Dm['persona_profile'] = sprof.round(2).to_dict('index')
    Dm['persona_success_rate'] = sd.groupby('cluster').success.mean().round(3).to_dict()
    Dm['persona_size'] = sd.cluster.value_counts().to_dict()
    C['persona_names'] = {cid: LEARNER_PERSONAS[i][0] for cid, i in name_clusters(jprof, LEARNER_PERSONAS).items()}
    Dm['persona_names'] = {cid: WORKSTYLE_PERSONAS[i][0] for cid, i in name_clusters(sprof, WORKSTYLE_PERSONAS).items()}

    models = {'j': j, 's': s, 'jm': jm, 'jsc': jsc, 'sm': sm, 'ssc': ssc,
              'jprof': jprof, 'jkm': jkm, 'jksc': jksc, 'sprof': sprof, 'skm': skm, 'sksc': sksc, 'jd': jd, 'sd': sd}
    return C, Dm, models
