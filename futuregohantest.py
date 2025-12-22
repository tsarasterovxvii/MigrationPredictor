#!/usr/bin/env python3
# PERFORMANCE-OPTIMIZED VERSION - Fixes the "Base features used" hang

import os
import math
import pickle
import time
from pathlib import Path
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pycountry

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

def safe_print(*a, **k):
    print(*a, **k, flush=True)

TOP_K_ARROWS = 5
FALLBACK_ISO3 = ["USA", "CAN", "GBR", "AUS", "NZL"]
DPRK_ISO3 = "PRK"

# ═══════════════════════════════════════════════════════════════════════════
# SKIP SLOW PARTS ON FIRST RUN - QUICK SETUP
# ═══════════════════════════════════════════════════════════════════════════

BILAT_FILE = r"C:\Users\Tsar Aster17\Downloads\bilat_mig.csv"
COUNTRIES_FILE = r"C:\Users\Tsar Aster17\Downloads\countries.csv"
CLEANED_DIR = r"C:\Users\Tsar Aster17\PycharmProjects\MigrationPredictor\cleaned"

CLEAN_GDP = os.path.join(CLEANED_DIR, "gdp_clean.csv")
CLEAN_GDP_PC = os.path.join(CLEANED_DIR, "gdp_percap_clean.csv")
CLEAN_POP = os.path.join(CLEANED_DIR, "population_clean.csv")
CLEAN_EDU = os.path.join(CLEANED_DIR, "education_clean.csv")
CLEAN_UNEMP = os.path.join(CLEANED_DIR, "unemployment_clean.csv")
CLEAN_EMP_SECTOR = os.path.join(CLEANED_DIR, "employment_clean.csv")

EXPORT_DIR = r"C:\Users\Tsar Aster17\PycharmProjects\MigrationPredictor\predictions"
os.makedirs(EXPORT_DIR, exist_ok=True)

CACHE_DIR = Path(".cache_migration")
CACHE_DIR.mkdir(exist_ok=True)

MODEL_FILE = CACHE_DIR / "single_migration_model.pkl"
MODEL_META_FILE = CACHE_DIR / "single_model_meta.pkl"
MACRO_MODELS_FILE = CACHE_DIR / "macro_models.pkl"
MACRO_FUTURE_FILE = CACHE_DIR / "macro_future_preds.pkl"
QUARTERLY_PRED_FILE = CACHE_DIR / "predictions_quarterly_2026_2050.pkl"
QUARTERLY_PRED_CSV = os.path.join(EXPORT_DIR, "predictions_quarterly_2026_2050.csv")

# QUICK EXIT: Check if everything is cached already
if (MODEL_FILE.exists() and MACRO_MODELS_FILE.exists() and
        MACRO_FUTURE_FILE.exists() and QUARTERLY_PRED_FILE.exists()):
    print("✓ All cache files exist. Skipping training and jumping to UI...")
    print("Loading cached files...")

    with open(MODEL_FILE, "rb") as f:
        model_pipe = pickle.load(f)
    with open(MODEL_META_FILE, "rb") as f:
        model_meta = pickle.load(f)
    with open(MACRO_MODELS_FILE, "rb") as f:
        macro_models = pickle.load(f)
    with open(MACRO_FUTURE_FILE, "rb") as f:
        macro_future_preds = pickle.load(f)
    with open(QUARTERLY_PRED_FILE, "rb") as f:
        all_predictions = pickle.load(f)

    print("✓ All caches loaded successfully!")

    # Load centroids for UI
    centroids = pd.read_csv(COUNTRIES_FILE)


    def iso2_to_iso3(iso2):
        if pd.isna(iso2):
            return None
        try:
            return pycountry.countries.get(alpha_2=str(iso2).strip()).alpha_3
        except Exception:
            return None


    centroids['ISO3'] = centroids['ISO'].apply(iso2_to_iso3)
    centroids = centroids.dropna(subset=['ISO3']).copy()
    centroids['ISO3'] = centroids['ISO3'].astype(str)

    country_coords = {}
    for _, r in centroids.iterrows():
        iso3 = str(r['ISO3'])
        try:
            country_coords[iso3] = (float(r['latitude']), float(r['longitude']))
        except Exception:
            continue

    print(f"✓ UI ready with {len(country_coords)} countries")

    # SKIP TO UI SECTION
    SKIP_TO_UI = True
else:
    SKIP_TO_UI = False
    print("Cache incomplete - running full training...")

if not SKIP_TO_UI:
    # Original code continues here
    START_YEAR = 2026
    END_YEAR = 2050
    QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
    TOP_K_ARROWS = 5
    FALLBACK_ISO3 = ["USA", "CAN", "GBR", "AUS", "NZL"]
    DPRK_ISO3 = "PRK"
    HGB_PARAMS = {"max_iter": 200, "max_depth": 6, "learning_rate": 0.05, "random_state": 42}
    DELTA_DAMPING = 0.6
    SMOOTHING_ALPHA = 0.35
    CLIP_N_SIGMA = 3.0
    SEASONAL_SCALE = 0.10


    def safe_print(*a, **k):
        print(*a, **k, flush=True)


    def iso2_to_iso3(iso2):
        if pd.isna(iso2):
            return None
        try:
            return pycountry.countries.get(alpha_2=str(iso2).strip()).alpha_3
        except Exception:
            return None


    def normalize_macro_df(df):
        dfc = df.copy()
        cols = list(dfc.columns)
        cols_low = [c.lower() for c in cols]

        if "country code" in cols_low or "countrycode" in cols_low:
            code_col = next((c for c in cols if c.lower() in ("country code", "countrycode")), None)
            yearcols = [c for c in cols if str(c).strip().isdigit()]
            if code_col and yearcols:
                long = dfc.melt(id_vars=[code_col], value_vars=yearcols, var_name="year", value_name="value")
                long = long.rename(columns={code_col: "ISO3"})
                long['year'] = pd.to_numeric(long['year'], errors='coerce')
                return long[['ISO3', 'year', 'value']].dropna(subset=['ISO3', 'year'])

        iso_candidates = [c for c in cols if c.lower() in ("iso3", "iso", "code", "country code", "country_code")]
        year_candidate = next((c for c in cols if c.lower() == 'year'), None)
        if iso_candidates and year_candidate:
            iso_col = iso_candidates[0]
            value_col = next(
                (c for c in cols if c not in (iso_col, year_candidate) and pd.api.types.is_numeric_dtype(dfc[c])), None)
            if value_col:
                out = dfc[[iso_col, year_candidate, value_col]].rename(
                    columns={iso_col: 'ISO3', year_candidate: 'year', value_col: 'value'})
                out['ISO3'] = out['ISO3'].astype(str)
                return out[['ISO3', 'year', 'value']].dropna(subset=['ISO3', 'year'])

        yearcols = [c for c in cols if str(c).strip().isdigit()]
        if yearcols and len(dfc.columns) > len(yearcols):
            id_col = [c for c in cols if c not in yearcols][0]
            long = dfc.melt(id_vars=[id_col], value_vars=yearcols, var_name='year', value_name='value')
            long = long.rename(columns={id_col: 'ISO3'})
            long['year'] = pd.to_numeric(long['year'], errors='coerce')
            return long[['ISO3', 'year', 'value']].dropna(subset=['ISO3', 'year'])

        return None


    # FAST COUNTRY CACHING
    def create_country_name_to_iso3_cache():
        cache = {}
        manual_mappings = {
            'United States': 'USA', 'US': 'USA', 'USA': 'USA',
            'United Kingdom': 'GBR', 'UK': 'GBR',
            'South Korea': 'KOR', 'Korea': 'KOR', 'North Korea': 'PRK',
            'Vietnam': 'VNM', 'Viet Nam': 'VNM',
            'Russia': 'RUS', 'Russian Federation': 'RUS',
            'Iran': 'IRN', 'Iran (Islamic Republic of)': 'IRN',
            'Syria': 'SYR', 'Syrian Arab Republic': 'SYR',
            'China': 'CHN', 'Mainland China': 'CHN', "People's Republic of China": 'CHN',
            'Taiwan': 'TWN', 'Hong Kong': 'HKG', 'Macau': 'MAC',
            'Congo': 'COG', 'Democratic Republic of Congo': 'COD', 'DRC': 'COD',
            "Cote d'Ivoire": 'CIV', 'Ivory Coast': 'CIV',
            'Turkey': 'TUR', 'Turkiye': 'TUR', 'Venezuela': 'VEN',
            'Egypt': 'EGY', 'Egypt (Arab Republic of)': 'EGY',
            'Yemen': 'YEM', 'Palestine': 'PSE', 'Palestinian Territory': 'PSE',
            'Bahamas': 'BHS', 'Bahamas, The': 'BHS',
            'Gambia': 'GMB', 'Gambia, The': 'GMB',
            'Moldova': 'MDA', 'Republic of Moldova': 'MDA',
        }

        for name, iso3 in manual_mappings.items():
            cache[name.upper().strip()] = iso3

        for country in pycountry.countries:
            cache[country.name.upper().strip()] = country.alpha_3
            if hasattr(country, 'official_name'):
                cache[country.official_name.upper().strip()] = country.alpha_3

        return cache


    def country_name_to_iso3_cached(name, cache):
        if pd.isna(name):
            return None
        name_clean = str(name).strip().upper()
        if name_clean in cache:
            return cache[name_clean]
        try:
            country = pycountry.countries.search_fuzzy(name)[0]
            iso3 = country.alpha_3
            cache[name_clean] = iso3
            return iso3
        except Exception:
            return None


    # LOAD DATA
    safe_print("Loading countries/centroids...")
    centroids = pd.read_csv(COUNTRIES_FILE)
    centroids['ISO3'] = centroids['ISO'].apply(iso2_to_iso3)
    centroids = centroids.dropna(subset=['ISO3']).copy()
    centroids['ISO3'] = centroids['ISO3'].astype(str)

    country_coords = {}
    for _, r in centroids.iterrows():
        iso3 = str(r['ISO3'])
        try:
            country_coords[iso3] = (float(r['latitude']), float(r['longitude']))
        except Exception:
            continue

    safe_print(f"✓ Mapped {len(country_coords)} countries with coords")

    safe_print("Loading bilateral migration data...")
    bilat = pd.read_csv(BILAT_FILE)
    safe_print(f"✓ bilat rows: {len(bilat)}")

    safe_print("Converting country names to ISO3 (caching makes this fast)...")
    country_cache = create_country_name_to_iso3_cache()
    bilat['orig'] = bilat['orig'].apply(lambda x: country_name_to_iso3_cached(x, country_cache))
    bilat['dest'] = bilat['dest'].apply(lambda x: country_name_to_iso3_cached(x, country_cache))
    bilat = bilat.dropna(subset=['orig', 'dest']).reset_index(drop=True)
    safe_print(f"✓ ISO3 mapping complete: {len(bilat)} rows")

    if 'year0' in bilat.columns:
        bilat['year0'] = pd.to_numeric(bilat['year0'], errors='coerce')
        bilat = bilat[bilat['year0'] >= 2005].reset_index(drop=True)
        safe_print(f"✓ Filtered to 2005+: {len(bilat)} rows")

    # LOAD MACROS
    safe_print("Loading macros...")
    macro_paths = {
        'gdp': CLEAN_GDP, 'gdp_percap': CLEAN_GDP_PC, 'population': CLEAN_POP,
        'education': CLEAN_EDU, 'unemployment': CLEAN_UNEMP, 'employment_sector': CLEAN_EMP_SECTOR
    }

    macro_dfs = {}
    for name, path in macro_paths.items():
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False)
                norm = normalize_macro_df(df)
                if norm is not None:
                    norm['ISO3'] = norm['ISO3'].astype(str).str.upper().str.strip()
                    norm['year'] = pd.to_numeric(norm['year'], errors='coerce').astype(int)
                    norm = norm.dropna(subset=['ISO3', 'year']).copy()
                    macro_dfs[name] = norm
                    safe_print(f"✓ Loaded {name}: {len(norm)} rows")
            except Exception as e:
                safe_print(f"⚠ Skipped {name}: {e}")


    # COMPUTE DELTAS
    def compute_macro_deltas(df_macro):
        rows = []
        for iso, grp in df_macro.groupby('ISO3'):
            grp = grp.sort_values('year')
            vals = grp[['year', 'value']].to_numpy()
            for i in range(len(vals) - 1):
                y0, y1 = int(vals[i, 0]), int(vals[i + 1, 0])
                v0, v1 = vals[i, 1], vals[i + 1, 1]
                if pd.isna(v0) or pd.isna(v1):
                    continue
                year_diff = y1 - y0
                delta = (float(v1) - float(v0)) / year_diff
                pct = (delta / float(v0)) if v0 > 0 else np.nan
                rows.append((iso, y0, delta, pct, year_diff))

        if not rows:
            return pd.DataFrame(columns=['ISO3', 'year', 'delta', 'pctdelta', 'year_gap'])
        return pd.DataFrame(rows, columns=['ISO3', 'year', 'delta', 'pctdelta', 'year_gap'])


    macro_deltas = {}
    for name, df in macro_dfs.items():
        macro_deltas[name] = compute_macro_deltas(df)
        safe_print(f"✓ Deltas for {name}: {len(macro_deltas[name])} rows")

    # TRAIN MACRO MODELS
    if MACRO_MODELS_FILE.exists() and MACRO_FUTURE_FILE.exists():
        safe_print("✓ Loading cached macro models...")
        with open(MACRO_MODELS_FILE, "rb") as f:
            macro_models = pickle.load(f)
        with open(MACRO_FUTURE_FILE, "rb") as f:
            macro_future_preds = pickle.load(f)
    else:
        safe_print("Training macro RFs...")
        macro_models = {}
        macro_future_preds = {}

        for name, df in macro_dfs.items():
            macro_models[name] = {}
            preds = {}

            for iso, grp in df.groupby('ISO3'):
                grp = grp.sort_values('year')
                if len(grp) >= 3:
                    X = grp[['year']].values
                    y = grp['value'].values
                    years = grp['year'].values.astype(float)
                    denom = float(max(1.0, (years.max() - years.min())))
                    w = 0.5 + (years - years.min()) / denom

                    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    try:
                        rf.fit(X, y, sample_weight=w)
                        macro_models[name][iso] = rf
                    except Exception:
                        pass

            for iso in centroids['ISO3'].unique():
                for year in range(START_YEAR, END_YEAR + 1):
                    if iso in macro_models[name]:
                        try:
                            val = float(macro_models[name][iso].predict([[year]])[0])
                        except Exception:
                            val = np.nan
                    else:
                        last = df[df['ISO3'] == iso].sort_values('year', ascending=False)
                        val = float(last.iloc[0]['value']) if len(last) > 0 else np.nan
                    preds[(iso, year)] = val

            macro_future_preds[name] = preds
            safe_print(f"✓ Macro {name} trained")

        with open(MACRO_MODELS_FILE, "wb") as f:
            pickle.dump(macro_models, f)
        with open(MACRO_FUTURE_FILE, "wb") as f:
            pickle.dump(macro_future_preds, f)

    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIMIZED DELTA DATASET BUILDING (THIS WAS THE BOTTLENECK)
    # ═══════════════════════════════════════════════════════════════════════════

    safe_print("Building training dataset (OPTIMIZED)...")

    base_candidates = ['year0', 'sd_drop_neg', 'sd_rev_neg', 'da_min_open', 'da_min_closed', 'da_pb_closed']
    base_feats = [c for c in base_candidates if c in bilat.columns]
    safe_print(f"Base features: {base_feats}")

    # Pre-compute categoricals ONCE, not repeatedly
    bilat['orig_code'] = pd.factorize(bilat['orig'])[0]
    bilat['dest_code'] = pd.factorize(bilat['dest'])[0]
    if 'quarter_sin' not in bilat.columns:
        bilat['quarter_sin'] = 0.0
    if 'quarter_cos' not in bilat.columns:
        bilat['quarter_cos'] = 1.0

    # BUILD INDEX for fast macro lookups
    delta_index = {}
    for name, md in macro_deltas.items():
        delta_index[name] = {}
        for _, row in md.iterrows():
            key = (row['ISO3'], int(row['year']))
            delta_index[name][key] = row

    safe_print("Building delta rows...")
    t0 = time.time()
    delta_rows = []
    bilat_sorted = bilat.sort_values(['orig', 'dest', 'year0']).reset_index(drop=True)

    # FAST iteration with groupby
    for (orig, dest), grp in bilat_sorted.groupby(['orig', 'dest'], sort=False):
        grp_sorted = grp.sort_values('year0').reset_index(drop=True)
        if len(grp_sorted) < 2:
            continue

        for i in range(len(grp_sorted) - 1):
            r0 = grp_sorted.iloc[i]
            r1 = grp_sorted.iloc[i + 1]

            if pd.isna(r0.get('mig_rate')) or pd.isna(r1.get('mig_rate')):
                continue

            year_t = int(r0['year0'])

            # Check macro availability ONCE per pair
            all_macro_ok = True
            for mname in macro_deltas.keys():
                if (orig, year_t) not in delta_index[mname] or (dest, year_t) not in delta_index[mname]:
                    all_macro_ok = False
                    break

            if not all_macro_ok:
                continue

            # BUILD ROW
            row = {f: r0.get(f, np.nan) for f in base_feats}
            row.update({
                'orig': orig, 'dest': dest,
                'orig_code': r0['orig_code'], 'dest_code': r0['dest_code'],
                'quarter_sin': 0.0, 'quarter_cos': 1.0,
                'year0': r0['year0'],
                'target_delta_mig_rate': float(r1['mig_rate']) - float(r0['mig_rate']),
                'sample_year': int(r1['year0'])
            })

            # ADD MACRO FEATURES
            for mname in macro_deltas.keys():
                v_o = delta_index[mname].get((orig, year_t), None)
                v_d = delta_index[mname].get((dest, year_t), None)
                if v_o is not None and v_d is not None:
                    row[f"{mname}_orig_delta"] = float(v_o['delta'])
                    row[f"{mname}_orig_pctdelta"] = float(v_o['pctdelta']) if not pd.isna(v_o['pctdelta']) else 0.0
                    row[f"{mname}_dest_delta"] = float(v_d['delta'])
                    row[f"{mname}_dest_pctdelta"] = float(v_d['pctdelta']) if not pd.isna(v_d['pctdelta']) else 0.0

            delta_rows.append(row)

    elapsed = time.time() - t0
    safe_print(f"✓ Delta rows built in {elapsed:.1f}s: {len(delta_rows)} rows")

    if len(delta_rows) > 0:
        df_delta = pd.DataFrame(delta_rows)

        years = df_delta['sample_year'].astype(float)
        years_ago = years.max() - years
        df_delta['sample_weight'] = np.exp(-years_ago / 10.0)

        ratio = df_delta['sample_weight'].max() / df_delta['sample_weight'].min()
        safe_print(f"✓ Sample weight ratio: {ratio:.1f}x")

        macro_delta_cols = [c for c in df_delta.columns if c.endswith('_delta') or c.endswith('_pctdelta')]
        feature_cols = base_feats + ['orig_code', 'dest_code', 'quarter_sin', 'quarter_cos'] + macro_delta_cols
        feature_cols = list(dict.fromkeys(feature_cols))

        safe_print(f"✓ Training features: {len(feature_cols)}")
    else:
        raise RuntimeError("No delta rows - cannot train!")

    # CONTINUE WITH TRAINING (same as original)
    delta_std = float(df_delta['target_delta_mig_rate'].std()) if len(df_delta) > 1 else 0.0
    delta_mean = float(df_delta['target_delta_mig_rate'].mean()) if len(df_delta) > 0 else 0.0

    # SCALE CORRECTION
    scale_candidates = []
    if 'mig_count' in bilat.columns:
        for _, r in bilat_sorted.iterrows():
            try:
                if pd.isna(r.get('mig_rate')) or pd.isna(r.get('mig_count')) or pd.isna(r.get('year0')):
                    continue
                iso = r['orig']
                y = int(r['year0'])

                if 'population' not in macro_dfs or len(macro_dfs['population']) == 0:
                    continue

                pop_hist = macro_dfs['population'][(macro_dfs['population']['ISO3'] == iso) &
                                                   (macro_dfs['population']['year'] == y)]
                if len(pop_hist) == 0:
                    continue

                pop = float(pop_hist['value'].iloc[0])
                if pd.isna(pop) or float(pop) <= 0:
                    continue

                implied = float(r['mig_rate']) * float(pop)
                if implied <= 0:
                    continue

                actual = float(r['mig_count'])
                ratio = actual / implied

                if np.isfinite(ratio) and 0 < ratio < 1000:
                    scale_candidates.append(ratio)
            except Exception:
                continue

    if len(scale_candidates) >= 5:
        q1, q3 = np.percentile(scale_candidates, [25, 75])
        iqr = q3 - q1
        filtered = [x for x in scale_candidates if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]
        SCALE_CORRECTION = float(np.median(filtered)) if len(filtered) >= 3 else float(np.median(scale_candidates))
        safe_print(f"✓ SCALE_CORRECTION: {SCALE_CORRECTION}")
    else:
        SCALE_CORRECTION = 1.0
        safe_print(f"✓ SCALE_CORRECTION: {SCALE_CORRECTION} (default)")


    # TRAIN MODEL
    def load_model_meta():
        if MODEL_FILE.exists() and MODEL_META_FILE.exists():
            try:
                with open(MODEL_FILE, "rb") as f:
                    mdl = pickle.load(f)
                with open(MODEL_META_FILE, "rb") as f:
                    meta = pickle.load(f)
                return mdl, meta
            except Exception:
                pass
        return None, None


    def save_model_meta(mdl, feat, ts):
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(mdl, f)
        meta = {'feature_cols': list(feat), 'target_scale': int(ts)}
        with open(MODEL_META_FILE, "wb") as f:
            pickle.dump(meta, f)


    model_pipe, model_meta = load_model_meta()
    need_retrain = model_pipe is None or model_meta is None or set(model_meta.get('feature_cols', [])) != set(
        feature_cols)

    if need_retrain:
        safe_print("Training HGB model...")
        hgb = HistGradientBoostingRegressor(**HGB_PARAMS)
        model_pipe = make_pipeline(SimpleImputer(strategy='mean'), hgb)

        X = df_delta[feature_cols].fillna(0).astype('float32')
        y = df_delta['target_delta_mig_rate'].fillna(0).astype('float32').values
        sample_weight = df_delta['sample_weight'].astype('float32').values

        model_pipe.fit(X, y, histgradientboostingregressor__sample_weight=sample_weight)
        save_model_meta(model_pipe, feature_cols, 1)
        model_meta = {'feature_cols': feature_cols, 'target_scale': 1}
        safe_print("✓ Model trained")
    else:
        safe_print("✓ Model loaded from cache")

    trained_feature_order = model_meta.get('feature_cols', None)


    # POPULATION HELPERS
    def interp_population(iso, year, quarter_idx):
        try:
            p_y = macro_future_preds.get('population', {}).get((iso, year), np.nan)
            p_y1 = macro_future_preds.get('population', {}).get((iso, year + 1), np.nan)

            if pd.isna(p_y) and not pd.isna(p_y1):
                p_y = p_y1

            if pd.isna(p_y):
                country_hist = macro_dfs['population'][macro_dfs['population']['ISO3'] == iso]
                if len(country_hist) > 0:
                    p_y = float(country_hist['value'].iloc[-1])
                else:
                    global_median = macro_dfs['population']['value'].median()
                    p_y = float(global_median) if not pd.isna(global_median) else 1_000_000.0

            if pd.isna(p_y1):
                return float(p_y)

            frac = (quarter_idx - 1) / 4.0
            return float(p_y + (p_y1 - p_y) * frac)
        except Exception:
            try:
                return float(macro_dfs['population']['value'].median())
            except Exception:
                return 1_000_000.0


    def stabilize_predicted_delta(pred_d, global_std=0.0, mean=0.0):
        d = float(pred_d) * float(DELTA_DAMPING)
        if global_std > 0:
            limit = CLIP_N_SIGMA * float(global_std)
            if d > mean + limit:
                d = mean + limit
            if d < mean - limit:
                d = mean - limit
        return float(d)


    # GENERATE PREDICTIONS
    if QUARTERLY_PRED_FILE.exists():
        safe_print("✓ Loading cached predictions...")
        with open(QUARTERLY_PRED_FILE, "rb") as f:
            all_predictions = pickle.load(f)
    else:
        safe_print("Generating quarterly predictions...")
        all_predictions = {}

        state = {}
        for iso in centroids['ISO3'].unique():
            iso = str(iso)
            s = {}
            p = macro_future_preds.get('population', {}).get((iso, START_YEAR), np.nan)
            if pd.isna(p):
                p = 1_000_000.0
            s['population'] = float(p)
            s['gdp'] = float(macro_future_preds.get('gdp', {}).get((iso, START_YEAR), np.nan) or 0.0)
            s['gdp_percap'] = float(macro_future_preds.get('gdp_percap', {}).get((iso, START_YEAR), np.nan) or (
                        s['gdp'] / max(1.0, s['population'])))
            s['education'] = float(macro_future_preds.get('education', {}).get((iso, START_YEAR), 0.0) or 0.0)
            s['unemployment'] = float(macro_future_preds.get('unemployment', {}).get((iso, START_YEAR), 0.0) or 0.0)
            s['employment_sector'] = float(
                macro_future_preds.get('employment_sector', {}).get((iso, START_YEAR), 0.0) or 0.0)
            state[iso] = s

        mig_rate_state = {}
        last_by_pair = {}

        for _, r in bilat_sorted.iterrows():
            pair = (r['orig'], r['dest'])
            yr = int(r['year0']) if not pd.isna(r.get('year0')) else -9999
            if pair not in last_by_pair or yr > last_by_pair[pair][0]:
                last_by_pair[pair] = (yr, r.get('mig_rate', 0.0))

        for pair, (yr, rate) in last_by_pair.items():
            mig_rate_state[pair] = float(rate if not pd.isna(rate) else 0.0)

        for _, r in bilat_sorted[['orig', 'dest']].drop_duplicates().iterrows():
            pair = (r['orig'], r['dest'])
            if pair not in mig_rate_state:
                mig_rate_state[pair] = 0.0

        bilat_static = bilat[['orig', 'dest', 'orig_code', 'dest_code']].copy()
        macro_keys = list(macro_future_preds.keys()) if isinstance(macro_future_preds, dict) else []

        for year in range(START_YEAR, END_YEAR + 1):
            for qidx, quarter in enumerate(["Q1", "Q2", "Q3", "Q4"], start=1):
                key = f"{quarter} {year}"
                safe_print(f"Predicting {key}...")

                df_future = bilat_static.copy()

                for feat in base_feats:
                    df_future[feat] = bilat.get(feat, 0.0)

                df_future['year0'] = year
                df_future['quarter'] = qidx
                df_future['quarter_sin'] = math.sin(2 * math.pi * (qidx / 4.0))
                df_future['quarter_cos'] = math.cos(2 * math.pi * (qidx / 4.0))

                for mname in macro_keys:
                    col_future = f"{mname}_orig_future"
                    col_baseline = f"{mname}_orig_baseline"
                    vals_f = []
                    vals_b = []

                    for iso in df_future['orig'].values:
                        iso = str(iso)
                        baseline_val = macro_future_preds.get(mname, {}).get((iso, year), np.nan)
                        vals_b.append(baseline_val)

                        v = state.get(iso, {}).get(mname, baseline_val)
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            v = baseline_val
                        vals_f.append(v)

                    df_future[col_future] = vals_f
                    df_future[col_baseline] = vals_b

                if trained_feature_order is None:
                    Xf = df_future[[c for c in df_future.columns if c in feature_cols]].fillna(0).astype('float32')
                else:
                    for c in trained_feature_order:
                        if c not in df_future.columns:
                            df_future[c] = 0.0
                    Xf = df_future[trained_feature_order].fillna(0).astype('float32')

                preds_delta = model_pipe.predict(Xf)
                stabilized_deltas = np.array(
                    [stabilize_predicted_delta(pdx, global_std=delta_std, mean=delta_mean) for pdx in preds_delta],
                    dtype=float)

                reconstructed_rates = []
                reconstructed_counts = []

                for idx, (orig, dest, pred_d) in enumerate(
                        zip(df_future['orig'].values, df_future['dest'].values, stabilized_deltas)):
                    orig = str(orig);
                    dest = str(dest)
                    pair_key = (orig, dest)

                    base_rate = float(mig_rate_state.get(pair_key, 0.0))
                    pred_rate_raw = base_rate + float(pred_d)

                    max_quarterly_rate = 0.05 / 4
                    pred_rate_raw_clipped = np.clip(pred_rate_raw, -max_quarterly_rate, max_quarterly_rate)

                    seasonal_adj = 1.0 + SEASONAL_SCALE * float(df_future.iloc[idx]['quarter_sin'])
                    pred_rate = pred_rate_raw_clipped * seasonal_adj

                    max_with_seasonality = max_quarterly_rate * 1.1
                    pred_rate = np.clip(pred_rate, -max_with_seasonality, max_with_seasonality)

                    pop = interp_population(orig, year, qidx)
                    cnt = pred_rate * pop
                    cnt_nonneg = max(0.0, cnt)

                    reconstructed_rates.append(pred_rate)
                    reconstructed_counts.append(cnt_nonneg)

                    prev_state = mig_rate_state.get(pair_key, 0.0)
                    smoothed = SMOOTHING_ALPHA * pred_rate + (1.0 - SMOOTHING_ALPHA) * prev_state
                    mig_rate_state[pair_key] = float(smoothed)

                df_future['predicted_mig_rate'] = np.array(reconstructed_rates, dtype=float)
                df_future['predicted_mig_count'] = np.array(reconstructed_counts, dtype=float)

                for iso in list(state.keys()):
                    p_interp = interp_population(iso, year, qidx)
                    state[iso]['population'] = float(p_interp)
                    state[iso]['gdp'] = float(
                        macro_future_preds.get('gdp', {}).get((iso, year), state[iso].get('gdp', 0.0)) or state[
                            iso].get('gdp', 0.0))
                    state[iso]['gdp_percap'] = float(
                        macro_future_preds.get('gdp_percap', {}).get((iso, year), state[iso].get('gdp_percap', 0.0)) or
                        state[iso].get('gdp_percap', 0.0))
                    state[iso]['unemployment'] = float(macro_future_preds.get('unemployment', {}).get((iso, year),
                                                                                                      state[iso].get(
                                                                                                          'unemployment',
                                                                                                          0.0)) or
                                                       state[iso].get('unemployment', 0.0))

                df_out = df_future[['orig', 'dest', 'predicted_mig_rate', 'predicted_mig_count']].copy()
                df_out = df_out[~((df_out['orig'] == DPRK_ISO3) | (df_out['dest'] == DPRK_ISO3))].reset_index(drop=True)

                all_predictions[key] = df_out

        with open(QUARTERLY_PRED_FILE, "wb") as f:
            pickle.dump(all_predictions, f)
        safe_print("✓ Predictions saved")

        if not os.path.exists(QUARTERLY_PRED_CSV):
            rows = []
            for k, dfq in all_predictions.items():
                q, y = k.split()
                temp = dfq.copy()
                temp['quarter'] = q
                temp['year'] = int(y)
                rows.append(temp)

            if rows:
                combined = pd.concat(rows, ignore_index=True)
                combined.to_csv(QUARTERLY_PRED_CSV, index=False)
                safe_print(f"✓ CSV exported to {QUARTERLY_PRED_CSV}")


# ═════════════════════════════════════════════════════════════════════════════
# DASH UI (SAME FOR BOTH PATHS)
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_for_display(df_future, current_year):
    """Aggregates migration data safely without returning None."""
    try:
        outflow = df_future.groupby('orig')['predicted_mig_count'].sum()
    except Exception:
        outflow = pd.Series(dtype=float)

    try:
        inflow = df_future.groupby('dest')['predicted_mig_count'].sum()
    except Exception:
        inflow = pd.Series(dtype=float)

    all_isos = set(outflow.index) | set(inflow.index)
    if not all_isos:
        empty = pd.DataFrame({
            'net_mig_count': pd.Series(dtype=float),
            'total_outflow_count': pd.Series(dtype=float),
            'population': pd.Series(dtype=float),
            'predicted_mig_pct': pd.Series(dtype=float),
            'norm': pd.Series(dtype=float)
        })
        return empty

    rows = []
    for iso in sorted(all_isos):
        o = outflow.get(iso, 0.0) if iso in outflow.index else 0.0
        i = inflow.get(iso, 0.0) if iso in inflow.index else 0.0
        net = i - o

        pop = 10_000_000.0
        if 'macro_dfs' in globals():
            try:
                if macro_dfs and 'population' in macro_dfs:
                    hist_pop = macro_dfs['population']
                    country_data = hist_pop[hist_pop['ISO3'] == iso]
                    if len(country_data) > 0:
                        pop = float(country_data['value'].iloc[-1])
            except Exception:
                pass

        pop = max(100_000.0, min(pop, 1_500_000_000.0))
        if iso in ["CHN", "IND"]:
            pct = (net / pop) / 80.0  # Divided by 20, then converted to %
        else:
            pct = (net / pop) / 10.0

        rows.append(
            {'ISO3': iso, 'net_mig_count': float(net), 'total_outflow_count': float(o), 'population': float(pop),
             'predicted_mig_pct': float(pct)})

    agg = pd.DataFrame(rows)

    if len(agg) > 0:
        agg['norm'] = (agg['predicted_mig_pct'] / 40.0)  # -100→0, 0→0.5, +100→1
        agg['norm'] = np.clip(agg['norm'], -20.0, 20.0)
    else:
        agg['norm'] = 0.0

    agg = agg.set_index('ISO3')
    return agg[['net_mig_count', 'total_outflow_count', 'population', 'predicted_mig_pct', 'norm']]


print("Starting Dash app...")
app = Dash(__name__)
app.title = "Global Migration Predictor"

quarter_options = sorted(list(all_predictions.keys()),
                         key=lambda k: (int(k.split()[1]), ["Q1", "Q2", "Q3", "Q4"].index(k.split()[0])))

app.layout = html.Div([
    html.H2("Global Migration Predictor", style={'textAlign': 'center'}),
    html.Div("Slide 2026→2050. Click countries to lock flows. Red implies migration out of the country and blue implies migration into the country.", style={'textAlign': 'center'}),
    dcc.Slider(
        id='quarter-slider',
        min=0, max=max(0, len(quarter_options) - 1), value=0,
        marks={i: quarter_options[i] for i in
               range(0, len(quarter_options), max(1, len(quarter_options) // 12))} if len(quarter_options) > 0 else {},
        step=1
    ),

    dcc.Store(id='clicked-countries', data=[]),
    dcc.Graph(id='migration-map', style={'height': '85vh'})
], style={'margin': '8px'})


@app.callback(
    Output('clicked-countries', 'data'),
    Input('migration-map', 'clickData'),
    Input('migration-map', 'relayoutData'),  # ← ADD THIS
    State('clicked-countries', 'data')
)
def toggle_click(clickData, relayoutData, clicked):
    if relayoutData and 'hover' in str(relayoutData):  # Mouse out
        return []  # Clear hover
    if clickData is None:
        return clicked

    try:
        pt = clickData['points'][0]
        cid = pt.get('location') or pt.get('text')
        if not cid: return clicked
        cid = str(cid).strip()
        if cid in clicked:
            clicked.remove(cid)
        else:
            clicked.append(cid)
        return clicked
    except:
        return clicked

def find_country_coords(code_or_name):
    if pd.isna(code_or_name) or code_or_name is None:
        return None
    c = str(code_or_name).strip()
    if c == DPRK_ISO3:
        return None
    if c in country_coords:
        return country_coords[c]
    return None


def _get_top5_for_origin(df_future, origin_iso):
    origin_iso = str(origin_iso)
    if origin_iso == DPRK_ISO3:
        return [fb for fb in FALLBACK_ISO3 if fb != origin_iso][:TOP_K_ARROWS]

    group = df_future[df_future['orig'] == origin_iso].copy()
    picks = []

    if group.shape[0] > 0:
        sorted_g = group.sort_values('predicted_mig_count', ascending=False)
        for _, r in sorted_g.iterrows():
            d = str(r['dest'])
            if d == origin_iso and d not in picks:
                picks.append(d)
            if len(picks) >= TOP_K_ARROWS:
                break

    for fb in FALLBACK_ISO3:
        if len(picks) >= TOP_K_ARROWS:
            break
        if fb == origin_iso or fb in picks:
            continue
        picks.append(fb)

    return picks[:TOP_K_ARROWS]


@app.callback(
    Output('migration-map', 'figure'),
    Input('quarter-slider', 'value'),
    Input('migration-map', 'hoverData'),
    State('clicked-countries', 'data')
)
def update_map(slider_index, hoverData, clicked_countries):
    if len(quarter_options) == 0:
        fig = go.Figure()
        fig.update_layout(title="No predictions")
        return fig

    slider_index = int(min(max(0, slider_index), len(quarter_options) - 1))
    quarter_key = quarter_options[slider_index]

    # Extract year from quarter key
    q, y = quarter_key.split()
    current_year = int(y)

    df_future = all_predictions[quarter_key].copy()
    df_future = df_future[~((df_future['orig'] == DPRK_ISO3) | (df_future['dest'] == DPRK_ISO3))].reset_index(drop=True)

    agg = aggregate_for_display(df_future, current_year)

    # FIXED: Correct column selection
    merged = centroids.set_index('ISO3').join(agg, how='left').reset_index()
    merged.loc[merged['ISO3'] == DPRK_ISO3, ['predicted_mig_pct', 'norm', 'net_mig_count']] = (None, None, None)

    locations = merged['ISO3'].tolist()
    zvals = merged['predicted_mig_pct'].fillna(0).tolist()

    customdata = np.vstack([
        np.zeros(len(merged)),
        merged['predicted_mig_pct'].fillna(0).astype('float32').tolist()
    ]).T

    textvals = merged.get('COUNTRY', merged['ISO3']).tolist()
    colorscale_custom = [[0, '#1C2663'], [0.25, '#1429B8'], [0.5, '#851DA8'], [0.75, '#DB0F0F'], [1, '#A81D1D']]

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=locations,
        z=zvals,
        zmin=-10,
        zmax=10,
        colorbar=dict(
            title='Net migration (%)',
            tickvals=[-10, 10],
            ticktext=['-10', '10']),
        text=textvals,  # textvals = country names
        customdata=customdata,  # customdata[:, 1] = migration percentage
        colorscale=colorscale_custom,
        colorbar_title='Predicted net migration (%)',
        marker_line_color='black',
        locationmode='ISO-3',
        hovertemplate='%{text}<br>Migration: %{customdata[1]:.2f}%<extra></extra>'
    ))
    clicked_active = set(clicked_countries or [])  # PERMANENT clicks only
    active = clicked_active.copy()  # Start with clicked countries

    # TEMPORARY hover (doesn't persist)
    if hoverData and 'points' in hoverData:
        try:
            hp = hoverData['points'][0]
            hover_iso = hp.get('location') or hp.get('text')
            if hover_iso and hover_iso not in clicked_active:  # Only if NOT already clicked
                active.add(str(hover_iso))
        except Exception:
            pass

    for origin in active:
        origin_coords = find_country_coords(origin)
        if not origin_coords:
            continue

        lat0, lon0 = origin_coords
        normv = float(agg['norm'].get(origin, 0.0)) if origin in agg.index else 0.0
        color_alpha = 0.25 + 0.6 * normv

        for d in _get_top5_for_origin(df_future, origin):
            if d == DPRK_ISO3:
                continue
            dest_coords = find_country_coords(d)
            if not dest_coords:
                continue
            lat1, lon1 = dest_coords
            fig.add_trace(go.Scattergeo(
                lon=[lon0, lon1], lat=[lat0, lat1], mode='lines',
                line=dict(width=2, color=f"rgba(0,0,0,0.5)"),
                hoverinfo='skip', showlegend=False
            ))

    fig.update_geos(showcoastlines=True, showcountries=True, showframe=False)
    fig.update_layout(title=f"{quarter_key}", geo=dict(projection_type='natural earth'),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


if __name__ == "__main__":
    print("✓ App ready at http://127.0.0.1:8050")
    try:
        app.run_server(debug=False, port=8050)
    except Exception:
        app.run(debug=False)