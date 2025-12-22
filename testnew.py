# appcode.py
# Full integrated stacked migration predictor + dynamic macro<->migration coupling + Dash UI
# Save and run: python appcode.py
# Requirements:
# pip install pandas numpy scikit-learn xgboost lightgbm plotly dash pycountry

import os
import math
import pickle
import time
from pathlib import Path
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
import pycountry

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor

import xgboost as xgb

# Dash / Plotly
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

# -------------------------
# USER PATHS - update if needed
# -------------------------
BILAT_FILE = r"C:\Users\Tsar Aster17\Downloads\bilat_mig.csv"
COUNTRIES_FILE = r"C:\Users\Tsar Aster17\Downloads\countries.csv"

# Optional cleaned macro CSVs (provide cleaned World Bank or OWID style files)
CLEANED_DIR = r"C:\Users\Tsar Aster17\PycharmProjects\MigrationPredictor\cleaned"
CLEAN_GDP = os.path.join(CLEANED_DIR, "gdp_clean.csv")
CLEAN_GDP_PC = os.path.join(CLEANED_DIR, "gdp_percap_clean.csv")
CLEAN_POP = os.path.join(CLEANED_DIR, "population_clean.csv")
CLEAN_EDU = os.path.join(CLEANED_DIR, "education_clean.csv")
CLEAN_UNEMP = os.path.join(CLEANED_DIR, "unemployment_clean.csv")
CLEAN_EMP_SECTOR = os.path.join(CLEANED_DIR, "employment_clean.csv")

# Cache / artifacts
CACHE_DIR = Path(".cache_migration")
CACHE_DIR.mkdir(exist_ok=True)
STACKED_MODEL_FILE = CACHE_DIR / "stacked_migration_model.pkl"
MODEL_META_FILE = CACHE_DIR / "stacked_model_meta.pkl"   # metadata (feature names + target scale + note that model predicts delta)
MACRO_MODELS_FILE = CACHE_DIR / "macro_models.pkl"
MACRO_FUTURE_FILE = CACHE_DIR / "macro_future_preds.pkl"
QUARTERLY_PRED_PKL = CACHE_DIR / "predictions_quarterly_2026_2050.pkl"
QUARTERLY_PRED_CSV = CACHE_DIR / "predictions_quarterly_2026_2050.csv"

# Simulation horizon
START_YEAR = 2026
END_YEAR = 2050
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

# UI / behavior
TOP_K_ARROWS = 5
FALLBACK_ISO3 = ["USA", "CAN", "GBR", "AUS", "NZL"]  # in order to fill up to top-k

# Stacking
STACK_CV = 3

# -------------------------
# Utilities
# -------------------------
def safe_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def iso2_to_iso3(iso2):
    if pd.isna(iso2): return None
    try:
        return pycountry.countries.get(alpha_2=str(iso2).strip()).alpha_3
    except Exception:
        return None

def normalize_macro_df(df):
    """
    Normalize macro DataFrame into (ISO3, year, value) if possible.
    Returns DataFrame with columns ['ISO3','year','value'] or None
    """
    dfc = df.copy()
    cols = list(dfc.columns)
    cols_low = [c.lower() for c in cols]
    # World Bank wide format detection
    if "country code" in cols_low or "countrycode" in cols_low:
        code_col = None
        for c in cols:
            if c.lower() in ("country code","countrycode"):
                code_col = c
                break
        yearcols = [c for c in cols if str(c).strip().isdigit()]
        if code_col and yearcols:
            long = dfc.melt(id_vars=[code_col], value_vars=yearcols, var_name="year", value_name="value")
            long = long.rename(columns={code_col:"ISO3"})
            long['year'] = pd.to_numeric(long['year'], errors='coerce')
            return long[['ISO3','year','value']].dropna(subset=['ISO3','year'])
    # OWID / long
    iso_candidates = [c for c in cols if c.lower() in ("iso3","iso","code","country code","country_code")]
    year_candidate = next((c for c in cols if c.lower()=='year'), None)
    if iso_candidates and year_candidate:
        iso_col = iso_candidates[0]
        # find numeric value column
        value_col = next((c for c in cols if c not in (iso_col, year_candidate) and pd.api.types.is_numeric_dtype(dfc[c])), None)
        if value_col:
            out = dfc[[iso_col, year_candidate, value_col]].rename(columns={iso_col:'ISO3', year_candidate:'year', value_col:'value'})
            out['ISO3'] = out['ISO3'].astype(str)
            return out[['ISO3','year','value']].dropna(subset=['ISO3','year'])
    # fallback wide: pick non-year column as ISO, year columns are digit strings
    yearcols = [c for c in cols if str(c).strip().isdigit()]
    if yearcols and len(dfc.columns) > len(yearcols):
        id_col = [c for c in cols if c not in yearcols][0]
        long = dfc.melt(id_vars=[id_col], value_vars=yearcols, var_name='year', value_name='value')
        long = long.rename(columns={id_col:'ISO3'})
        long['year'] = pd.to_numeric(long['year'], errors='coerce')
        return long[['ISO3','year','value']].dropna(subset=['ISO3','year'])
    return None

# -------------------------
# 1) Load base datasets
# -------------------------
safe_print("Loading countries/centroids...")
if not os.path.exists(COUNTRIES_FILE):
    raise FileNotFoundError(f"countries file missing at {COUNTRIES_FILE}")
centroids = pd.read_csv(COUNTRIES_FILE)
safe_print("centroids columns:", list(centroids.columns))

if 'ISO' not in centroids.columns:
    raise ValueError("countries.csv must include 'ISO' column (alpha-2).")
if 'latitude' not in centroids.columns or 'longitude' not in centroids.columns:
    raise ValueError("countries.csv must include 'latitude' and 'longitude' columns.")

centroids['ISO3'] = centroids['ISO'].apply(iso2_to_iso3)
centroids = centroids.dropna(subset=['ISO3']).copy()
centroids['ISO3'] = centroids['ISO3'].astype(str)

# map ISO3 -> (lat, lon) (also keep a DataFrame lookup)
country_coords = {}
for _, r in centroids.iterrows():
    iso3 = str(r['ISO3'])
    try:
        country_coords[iso3] = (float(r['latitude']), float(r['longitude']))
    except Exception:
        continue
safe_print(f"Mapped {len(country_coords)} countries with coords")

# bilateral migration
safe_print("Loading bilateral migration data...")
if not os.path.exists(BILAT_FILE):
    raise FileNotFoundError(f"bilat file missing at {BILAT_FILE}")
bilat = pd.read_csv(BILAT_FILE)
safe_print("bilat columns:", list(bilat.columns))
safe_print("bilat rows:", len(bilat))

if 'orig' not in bilat.columns or 'dest' not in bilat.columns:
    raise ValueError("bilat CSV must include 'orig' and 'dest' columns.")
bilat['orig'] = bilat['orig'].astype(str).str.upper().str.strip()
bilat['dest'] = bilat['dest'].astype(str).str.upper().str.strip()

# -------------------------
# 2) Load cleaned macro CSVs (if present) and normalize
# -------------------------
safe_print("Loading cleaned macros (if available)...")
macro_paths = {
    'gdp': CLEAN_GDP,
    'gdp_percap': CLEAN_GDP_PC,
    'population': CLEAN_POP,
    'education': CLEAN_EDU,
    'unemployment': CLEAN_UNEMP,
    'employment_sector': CLEAN_EMP_SECTOR
}
macro_dfs = {}
for name, path in macro_paths.items():
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path, low_memory=False)
            norm = normalize_macro_df(df)
            if norm is None:
                safe_print(f"Warning: Could not normalize {path}; skipping {name}")
                continue
            norm['ISO3'] = norm['ISO3'].astype(str).str.upper().str.strip()
            norm['year'] = pd.to_numeric(norm['year'], errors='coerce').astype(int)
            norm = norm.dropna(subset=['ISO3','year']).copy()
            macro_dfs[name] = norm
            safe_print(f"Loaded macro '{name}' rows: {len(norm)}")
        except Exception as e:
            safe_print(f"Failed to load macro {name} from {path}: {e}")
    else:
        safe_print(f"Macro file not found for {name}: {path}")

# -------------------------
# Helper: compute recency weights for years (older -> lower)
# -------------------------
def compute_recency_weights(years, min_w=0.5, max_w=1.5):
    yrs = np.array(years, dtype=float)
    yrmin = yrs.min()
    yrmax = yrs.max()
    rng = max(1.0, yrmax - yrmin)
    w = min_w + (yrs - yrmin) / rng * (max_w - min_w)
    return w

# -------------------------
# 3) Train macro forecasters (per-country RF) and produce base future forecasts (cached)
# -------------------------
if MACRO_FUTURE_FILE.exists() and MACRO_MODELS_FILE.exists():
    safe_print("Loading cached macro models & future preds...")
    with open(MACRO_MODELS_FILE, "rb") as f:
        macro_models = pickle.load(f)
    with open(MACRO_FUTURE_FILE, "rb") as f:
        macro_future_preds = pickle.load(f)
else:
    safe_print("Training macro per-country forecasters and building future baseline forecasts (this may take a while)...")
    macro_models = {}
    macro_future_preds = {}
    for name, df in macro_dfs.items():
        safe_print("Processing macro:", name)
        macro_models[name] = {}
        preds = {}
        # train RF per country if enough historic points
        for iso, grp in df.groupby('ISO3'):
            grp = grp.sort_values('year')
            if len(grp) >= 3:
                # Train on year -> value but weight recent years higher
                X = grp[['year']].values
                y = grp['value'].values
                years = grp['year'].values
                w = compute_recency_weights(years, min_w=0.6, max_w=1.6)
                rf = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
                try:
                    rf.fit(X, y, sample_weight=w)
                    macro_models[name][iso] = rf
                except Exception:
                    # fallback to unweighted
                    try:
                        rf.fit(X, y)
                        macro_models[name][iso] = rf
                    except Exception:
                        continue
        # for each iso and year in horizon produce baseline forecast (model or last observed)
        for iso in list(centroids['ISO3'].unique()):
            for year in range(START_YEAR, END_YEAR+1):
                if iso in macro_models[name]:
                    try:
                        val = float(macro_models[name][iso].predict([[year]])[0])
                    except Exception:
                        val = np.nan
                else:
                    last = df[df['ISO3']==iso].sort_values('year', ascending=False)
                    if len(last) > 0:
                        val = float(last.iloc[0]['value'])
                    else:
                        val = np.nan
                preds[(iso, year)] = val
        macro_future_preds[name] = preds
        safe_print(f"Macro {name} baseline preds created (models for {len(macro_models[name])} countries).")
    # cache them
    with open(MACRO_MODELS_FILE, "wb") as f:
        pickle.dump(macro_models, f)
    with open(MACRO_FUTURE_FILE, "wb") as f:
        pickle.dump(macro_future_preds, f)
    safe_print("Saved macro models & future preds.")

# -------------------------
# 4) Prepare migration model features & delta-based training dataset
# -------------------------
safe_print("Preparing migration model features (delta-based)...")
print("Interpolating 5-year migration data to annual...")

bilat_sorted = bilat.sort_values(["orig", "dest", "year0"])
annual_rows = []

for (o, d), grp in bilat_sorted.groupby(["orig", "dest"]):
    grp = grp.sort_values("year0")
    for i in range(len(grp) - 1):
        y1 = int(grp.iloc[i]["year0"])
        y2 = int(grp.iloc[i + 1]["year0"])
        span = y2 - y1
        if span <= 1:
            continue

        for y in range(y1, y2 + 1):
            frac = (y - y1) / span
            row = grp.iloc[i].copy()
            row["year0"] = y

            for col in ["sd_drop_neg", "sd_rev_neg", "mig_rate",
                        "da_min_open", "da_min_closed", "da_pb_closed"]:
                row[col] = (
                        grp.iloc[i][col] * (1 - frac)
                        + grp.iloc[i + 1][col] * frac
                )
            annual_rows.append(row)

annual_df = pd.DataFrame(annual_rows)
print("Annual migration rows after interpolation:", len(annual_df))

bilat = annual_df
base_candidates = ['year0','sd_drop_neg','sd_rev_neg','da_min_open','da_min_closed','da_pb_closed']
base_feats = [c for c in base_candidates if c in bilat.columns]
safe_print("Base features used:", base_feats)

# ensure year0 present and numeric
if 'year0' not in bilat.columns:
    raise ValueError("bilat must include 'year0' column")
bilat['year0'] = pd.to_numeric(bilat['year0'], errors='coerce').astype(int)

# categorical codes for origin/destination
bilat['orig_code'] = bilat['orig'].astype('category').cat.codes
bilat['dest_code'] = bilat['dest'].astype('category').cat.codes

# Add seasonal placeholders for historical rows
if 'quarter_sin' not in bilat.columns:
    bilat['quarter_sin'] = 0.0
if 'quarter_cos' not in bilat.columns:
    bilat['quarter_cos'] = 1.0

# Build a lookup for macros by (iso, year)
macro_lookup = {}
for m, df in macro_dfs.items():
    macro_lookup[m] = {(r.ISO3, int(r.year)): r.value for r in df.itertuples(index=False)}

# Build delta training rows:
# For each (orig,dest,year) where both current and previous year rows exist (for same orig,dest),
# compute target = mig_rate_curr - mig_rate_prev
# features: base feats for current row + orig_code/dest_code + macro deltas for orig & dest (curr - prev) and pct deltas
rows = []
weights = []
# We'll track most recent year in data for weighting
all_years = bilat['year0'].unique()
if len(all_years) == 0:
    raise ValueError("No year0 values found in bilat")
year_now = max(all_years)

# prepare an index for quick lookup by (orig,dest,year)
bilat_index = {(r.orig, r.dest, int(r.year0)): r for r in bilat.itertuples(index=False)}

for r in bilat.itertuples(index=False):
    orig = r.orig
    dest = r.dest
    year = int(r.year0)
    prev_key = (orig, dest, year-1)
    cur_key = (orig, dest, year)
    if prev_key not in bilat_index:
        continue
    prev_row = bilat_index[prev_key]
    cur_row = bilat_index[cur_key]
    # require both mig_rate present
    try:
        mig_prev = float(prev_row.mig_rate)
        mig_curr = float(cur_row.mig_rate)
    except Exception:
        continue
    # compute target as delta (current - prev). We will train model to predict delta.
    target_delta = mig_curr - mig_prev
    # gather base features from current row
    feat = {}
    feat['year0'] = year
    for f in base_feats:
        feat[f] = getattr(cur_row, f) if hasattr(cur_row, f) else np.nan
    feat['orig_code'] = getattr(cur_row, 'orig_code')
    feat['dest_code'] = getattr(cur_row, 'dest_code')
    # quarter placeholders (neutral historically)
    feat['quarter_sin'] = 0.0
    feat['quarter_cos'] = 1.0
    # add macro deltas (orig & dest)
    missing_macro = False
    for mname in macro_paths.keys():
        # current macro for orig (year)
        curr_val_o = macro_lookup.get(mname, {}).get((orig, year), np.nan)
        prev_val_o = macro_lookup.get(mname, {}).get((orig, year-1), np.nan)
        curr_val_d = macro_lookup.get(mname, {}).get((dest, year), np.nan)
        prev_val_d = macro_lookup.get(mname, {}).get((dest, year-1), np.nan)
        # If both current and previous are missing for an iso, skip this row (user requested dropping rows with missing deltas)
        if (pd.isna(curr_val_o) and pd.isna(prev_val_o)) or (pd.isna(curr_val_d) and pd.isna(prev_val_d)):
            missing_macro = True
            break
        # set defaults if one side available
        # Only compute delta if both available; if one missing, mark missing and break (drop)
        if pd.isna(curr_val_o) or pd.isna(prev_val_o) or pd.isna(curr_val_d) or pd.isna(prev_val_d):
            missing_macro = True
            break
        # compute deltas
        delta_o = curr_val_o - prev_val_o
        delta_d = curr_val_d - prev_val_d
        pct_o = (delta_o / prev_val_o) if prev_val_o != 0 else 0.0
        pct_d = (delta_d / prev_val_d) if prev_val_d != 0 else 0.0
        feat[f"{mname}_orig_delta"] = float(delta_o)
        feat[f"{mname}_orig_pctdelta"] = float(pct_o)
        feat[f"{mname}_dest_delta"] = float(delta_d)
        feat[f"{mname}_dest_pctdelta"] = float(pct_d)
    if missing_macro:
        continue
    # append row
    rows.append((feat, target_delta))
    # weight sample by recency relative to latest year (closer to year_now gets higher weight)
    # weight = 0.5 .. 1.5 scaled linearly
    sample_w = 0.5 + (year - min(all_years)) / max(1, (year_now - min(all_years)))  # 0.5..1.5
    weights.append(sample_w)

safe_print(f"Prepared delta-based training rows: {len(rows)}  (dropped {len(bilat) - len(rows)} rows due to missing deltas/targets)")

# If we have too few rows, don't crash — train will still run but with warning
# Build training DataFrame
if len(rows) > 0:
    feat_dicts = [r[0] for r in rows]
    y_vals = np.array([r[1] for r in rows], dtype=float)
    X_df = pd.DataFrame(feat_dicts)
else:
    X_df = pd.DataFrame(columns=['year0'])
    y_vals = np.array([], dtype=float)

# final feature_cols (ordered)
feature_cols = []
# base features
feature_cols += [c for c in base_feats]
feature_cols += ['orig_code','dest_code','quarter_sin','quarter_cos']
# macro delta features (from one sample row if exists)
if len(X_df) > 0:
    for c in X_df.columns:
        if c not in feature_cols:
            feature_cols.append(c)
# ensure uniqueness
feature_cols = list(dict.fromkeys(feature_cols))
safe_print("Final feature columns (for training):", feature_cols)

# -------------------------
# 5) Build stacked ensemble and train (or load cached), with feature-name metadata handling
# Note: we removed Pipeline wrappers so sample_weight works correctly.
# -------------------------
def load_model_and_meta():
    """Load cached model and its metadata if both exist, else return (None, None)."""
    if STACKED_MODEL_FILE.exists() and MODEL_META_FILE.exists():
        try:
            with open(STACKED_MODEL_FILE, "rb") as f:
                mdl = pickle.load(f)
            with open(MODEL_META_FILE, "rb") as f:
                meta = pickle.load(f)
            return mdl, meta
        except Exception as e:
            safe_print("Warning: failed to load cached model or meta:", e)
            return None, None
    return None, None

def save_model_and_meta(mdl, feature_list, target_scale_flag):
    with open(STACKED_MODEL_FILE, "wb") as f:
        pickle.dump(mdl, f)
    feature_list_unique = list(dict.fromkeys(list(feature_list)))
    meta = {'feature_cols': feature_list_unique, 'target_scale': int(target_scale_flag), 'predicts': 'delta'}
    with open(MODEL_META_FILE, "wb") as f:
        pickle.dump(meta, f)

stack_model, model_meta = load_model_and_meta()

need_retrain = False
if stack_model is not None and model_meta is not None:
    trained_cols = model_meta.get('feature_cols', [])
    set_trained = set([str(c) for c in trained_cols])
    set_current = set([str(c) for c in feature_cols])
    if set_trained != set_current:
        safe_print("Cached stacked model feature set differs from current feature set. Will retrain model.")
        need_retrain = True
else:
    need_retrain = True

if need_retrain:
    safe_print("Building stacked ensemble (RF, XGB, HistGB, optional LGB) and training — this may be slow...")
    estimators = []
    # plain estimators (no Pipeline) so sample_weight works
    rf = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    estimators.append(('rf', rf))
    xgb_est = xgb.XGBRegressor(n_estimators=105, learning_rate=0.05, max_depth=5, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=1)
    estimators.append(('xgb', xgb_est))
    hgb = HistGradientBoostingRegressor(max_iter=200, random_state=42)
    estimators.append(('hgb', hgb))

    final_est = xgb.XGBRegressor(n_estimators=105, learning_rate=0.03, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=1)
    stack_model = StackingRegressor(estimators=estimators, final_estimator=final_est,
                                   cv=STACK_CV, n_jobs=-1, passthrough=False)

    # Prepare X and y, impute missing with SimpleImputer (do not drop columns)
    if len(X_df) == 0:
        raise RuntimeError("No training rows available after delta processing. Cannot train model.")
    # Reindex columns to feature_cols (add missing columns as zeros)
    for c in feature_cols:
        if c not in X_df.columns:
            X_df[c] = 0.0
    X_df = X_df[feature_cols].copy()
    # Impute (mean)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_df.values)
    # convert weights to numpy
    w_arr = np.array(weights, dtype=float)
    # Fit stacking regressor with sample_weight (we removed Pipeline wrappers so underlying estimators receive sample_weight)
    safe_print("Fitting stacked model on", X_imputed.shape[0], "rows and", X_imputed.shape[1], "features ...")
    stack_model.fit(X_imputed, y_vals, sample_weight=w_arr)
    # determine target_scale (if delta is large, probably percentages in original, remain conservative)
    median_abs_y = float(np.median(np.abs(y_vals)))
    target_scale = 100 if median_abs_y > 1.5 else 1
    save_model_and_meta(stack_model, feature_cols, target_scale)
    # Save imputer too (we'll need it for prediction)
    with open(CACHE_DIR / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    safe_print("Saved stacked model to", STACKED_MODEL_FILE, " (target_scale=", target_scale, ")")
else:
    safe_print("Loaded cached stacked migration model and metadata.")
    # load imputer
    try:
        with open(CACHE_DIR / "imputer.pkl", "rb") as f:
            imputer = pickle.load(f)
    except Exception:
        imputer = SimpleImputer(strategy='mean')  # fallback (shouldn't be needed)

# load meta
trained_feature_order = None
target_scale = 1
if MODEL_META_FILE.exists():
    try:
        with open(MODEL_META_FILE, "rb") as f:
            meta = pickle.load(f)
        trained_feature_order = meta.get('feature_cols', None)
        if trained_feature_order is not None:
            trained_feature_order = list(dict.fromkeys(trained_feature_order))
        target_scale = int(meta.get('target_scale', 1))
    except Exception:
        trained_feature_order = None
        target_scale = 1
safe_print("Model target_scale:", target_scale, "  (model predicts deltas)")

# -------------------------
# 6) Coupled dynamic simulation across quarters (Q1 2026 — Q4 2050)
# Use Option A: only regenerate if cached preds missing
# -------------------------
expected_keys = [f"{q} {y}" for y in range(START_YEAR, END_YEAR+1) for q in QUARTERS]
regen = False
if QUARTERLY_PRED_PKL.exists() and QUARTERLY_PRED_CSV.exists():
    safe_print("Found cached quarterly predictions (pkl + csv); loading...")
    with open(QUARTERLY_PRED_PKL, "rb") as f:
        all_predictions = pickle.load(f)
    # validate completeness
    missing = [k for k in expected_keys if k not in all_predictions]
    if missing:
        safe_print("Cached predictions incomplete; regenerating all forecasts.")
        regen = True
else:
    regen = True

if regen:
    safe_print("Starting dynamic coupled simulation and generating quarterly predictions — this may take significant time...")
    all_predictions = {}
    # initialize per-country current macro state from last observed values or baseline macro_future_preds for START_YEAR
    state = {}
    def macro_baseline(mname, iso, year):
        val = macro_future_preds.get(mname, {}).get((iso, year), np.nan) if isinstance(macro_future_preds, dict) else np.nan
        return val

    for iso in centroids['ISO3'].unique():
        iso = str(iso)
        s = {}
        # population
        pop_val = None
        if 'population' in macro_dfs:
            last = macro_dfs['population'][macro_dfs['population']['ISO3']==iso].sort_values('year', ascending=False)
            if len(last)>0: pop_val = float(last.iloc[0]['value'])
        if pop_val is None:
            pop_val = macro_baseline('population', iso, START_YEAR) or 1_000_000.0
        s['population'] = float(pop_val)
        # gdp
        gdp_val = None
        if 'gdp' in macro_dfs:
            last = macro_dfs['gdp'][macro_dfs['gdp']['ISO3']==iso].sort_values('year', ascending=False)
            if len(last)>0: gdp_val = float(last.iloc[0]['value'])
        if gdp_val is None:
            gdp_val = macro_baseline('gdp', iso, START_YEAR) or 1e9
        s['gdp'] = float(gdp_val)
        # gdp_percap
        if 'gdp_percap' in macro_dfs:
            last = macro_dfs['gdp_percap'][macro_dfs['gdp_percap']['ISO3']==iso].sort_values('year', ascending=False)
            if len(last)>0:
                s['gdp_percap'] = float(last.iloc[0]['value'])
            else:
                s['gdp_percap'] = s['gdp'] / max(1, s['population'])
        else:
            s['gdp_percap'] = s['gdp'] / max(1, s['population'])
        # other macros
        s['education'] = float(macro_baseline('education', iso, START_YEAR) or 0.0)
        s['unemployment'] = float(macro_baseline('unemployment', iso, START_YEAR) or 0.0)
        s['employment_sector'] = float(macro_baseline('employment_sector', iso, START_YEAR) or 0.0)
        state[iso] = s

    # Precompute static columns in bilat for building df_future
    static_columns = ['orig','dest','orig_code','dest_code']
    for col in static_columns:
        if col not in bilat.columns:
            bilat[col] = ""
    bilat_static = bilat[static_columns].copy()

    t0 = time.time()
    total_steps = (END_YEAR - START_YEAR + 1) * 4
    step_idx = 0

    # Prepare mapping of last observed mig_rate for each (orig,dest)
    last_mig_rate = {}
    bilat_sorted = bilat.sort_values('year0')
    for (orig, dest), grp in bilat_sorted.groupby(['orig','dest']):
        grp2 = grp.sort_values('year0', ascending=False)
        last_mig_rate[(orig, dest)] = float(grp2.iloc[0]['mig_rate']) if len(grp2)>0 else np.nan

    for year in range(START_YEAR, END_YEAR+1):
        for qidx, quarter in enumerate(QUARTERS, start=1):
            step_idx += 1
            key = f"{quarter} {year}"
            safe_print(f"Simulating {key} ({step_idx}/{total_steps}) ...")
            # Build df_future
            df_future = bilat_static.copy()
            for feat in base_feats:
                if feat in bilat.columns:
                    df_future[feat] = bilat[feat]
                else:
                    df_future[feat] = 0.0
            df_future['year0'] = year
            df_future['quarter'] = qidx
            df_future['quarter_sin'] = math.sin(2*math.pi*(qidx/4.0))
            df_future['quarter_cos'] = math.cos(2*math.pi*(qidx/4.0))
            # attach macro future & baseline for this year and also add dynamic state values
            for mname in macro_future_preds.keys():
                col_future = f"{mname}_orig_future"
                col_baseline = f"{mname}_orig_baseline"
                vals_f = []
                vals_b = []
                for iso in df_future['orig'].values:
                    iso = str(iso)
                    baseline_val = macro_future_preds.get(mname, {}).get((iso, year), np.nan)
                    vals_b.append(baseline_val)
                    s = state.get(iso, {})
                    v = s.get(mname) if mname in s else None
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        v = baseline_val
                    vals_f.append(v)
                df_future[col_future] = vals_f
                df_future[col_baseline] = vals_b

            # Build Xf with same features used in training (delta feature names)
            # We must compute macro deltas between year and year-1 for orig & dest so the model receives same schema
            macro_delta_cols = []
            for mname in macro_paths.keys():
                o_curr = []
                o_prev = []
                d_curr = []
                d_prev = []
                for orig, dest in zip(df_future['orig'].values, df_future['dest'].values):
                    orig = str(orig); dest = str(dest)
                    curr_o = macro_lookup.get(mname, {}).get((orig, year), np.nan)
                    prev_o = macro_lookup.get(mname, {}).get((orig, year-1), np.nan)
                    curr_d = macro_lookup.get(mname, {}).get((dest, year), np.nan)
                    prev_d = macro_lookup.get(mname, {}).get((dest, year-1), np.nan)
                    # fallback to baseline preds if historical missing
                    if pd.isna(curr_o): curr_o = macro_future_preds.get(mname, {}).get((orig, year), np.nan)
                    if pd.isna(prev_o): prev_o = macro_future_preds.get(mname, {}).get((orig, year-1), np.nan)
                    if pd.isna(curr_d): curr_d = macro_future_preds.get(mname, {}).get((dest, year), np.nan)
                    if pd.isna(prev_d): prev_d = macro_future_preds.get(mname, {}).get((dest, year-1), np.nan)
                    o_curr.append(curr_o); o_prev.append(prev_o); d_curr.append(curr_d); d_prev.append(prev_d)
                df_future[f"{mname}_orig_delta"] = [ (a - b) if (not pd.isna(a) and not pd.isna(b)) else 0.0 for a,b in zip(o_curr,o_prev) ]
                df_future[f"{mname}_orig_pctdelta"] = [ ((a-b)/b) if (not pd.isna(a) and not pd.isna(b) and b!=0) else 0.0 for a,b in zip(o_curr,o_prev) ]
                df_future[f"{mname}_dest_delta"] = [ (a - b) if (not pd.isna(a) and not pd.isna(b)) else 0.0 for a,b in zip(d_curr,d_prev) ]
                df_future[f"{mname}_dest_pctdelta"] = [ ((a-b)/b) if (not pd.isna(a) and not pd.isna(b) and b!=0) else 0.0 for a,b in zip(d_curr,d_prev) ]
                macro_delta_cols += [f"{mname}_orig_delta", f"{mname}_orig_pctdelta", f"{mname}_dest_delta", f"{mname}_dest_pctdelta"]

            # Combine desired columns but preserve trained feature order
            if trained_feature_order is not None:
                # ensure all trained_feature_order columns exist
                for c in trained_feature_order:
                    if c not in df_future.columns:
                        df_future[c] = 0.0
                Xf_df = df_future[trained_feature_order].copy()
            else:
                # fallback: use feature_cols
                for c in feature_cols:
                    if c not in df_future.columns:
                        df_future[c] = 0.0
                Xf_df = df_future[feature_cols].copy()

            # Impute
            Xf_arr = imputer.transform(Xf_df.fillna(0).values)

            # Predict delta
            preds_delta = stack_model.predict(Xf_arr)
            # interpret target_scale (if model trained with percentages)
            if target_scale == 100:
                preds_delta = np.array(preds_delta, dtype=float) / 100.0
            else:
                preds_delta = np.array(preds_delta, dtype=float)

            # Apply delta to baseline mig_rate (last observed for this pair if available)
            predicted_rates = []
            for orig, dest, delta in zip(df_future['orig'].values, df_future['dest'].values, preds_delta):
                baseline = last_mig_rate.get((orig, dest), np.nan)
                if pd.isna(baseline):
                    # fallback: treat baseline as 0
                    new_rate = delta
                else:
                    new_rate = baseline + delta
                predicted_rates.append(float(new_rate))
            df_future['predicted_mig_rate'] = np.array(predicted_rates, dtype=float)

            # clip absurd values
            df_future['predicted_mig_rate'] = df_future['predicted_mig_rate'].clip(lower=-1.0, upper=1.0)  # fraction bounds

            # season adj (tiny)
            season_adj = 1.0 + 0.02 * df_future['quarter_sin']
            df_future['predicted_mig_rate'] = df_future['predicted_mig_rate'] * season_adj

            # Convert to counts using current origin population state
            counts = []
            for iso, rate in zip(df_future['orig'].values, df_future['predicted_mig_rate'].values):
                iso = str(iso)
                pop = state.get(iso, {}).get('population', 1_000_000.0)
                cnt = rate * pop
                counts.append(max(0.0, cnt))
            df_future['predicted_mig_count'] = counts

            # Aggregate flows
            outflow = df_future.groupby('orig')['predicted_mig_count'].sum().to_dict()
            inflow = df_future.groupby('dest')['predicted_mig_count'].sum().to_dict()

            # Save net migration count per origin for hover
            net_by_origin = {}
            for iso in df_future['orig'].unique():
                out = outflow.get(iso, 0.0)
                inc = inflow.get(iso, 0.0)
                net_by_origin[iso] = inc - out

            # Update dynamic state
            POP_TO_GDP_ELASTICITY = 0.2
            MIGRATION_TO_POP_SCALER = 1.0
            for iso in list(state.keys()):
                prev_pop = state[iso]['population']
                out = outflow.get(iso, 0.0)
                inc = inflow.get(iso, 0.0)
                net_mig = inc - out
                new_pop = max(1.0, prev_pop + MIGRATION_TO_POP_SCALER * net_mig)
                state[iso]['population'] = new_pop
                baseline_gdp = macro_future_preds.get('gdp', {}).get((iso, year), np.nan) if 'gdp' in macro_future_preds else state[iso].get('gdp', np.nan)
                if math.isnan(baseline_gdp):
                    baseline_gdp = state[iso].get('gdp', 0.0)
                pop_change_frac = (new_pop - prev_pop) / prev_pop if prev_pop>0 else 0.0
                adjusted_gdp = baseline_gdp * (1.0 + POP_TO_GDP_ELASTICITY * pop_change_frac)
                state[iso]['gdp'] = adjusted_gdp
                state[iso]['gdp_percap'] = (adjusted_gdp / new_pop) if new_pop>0 else 0.0
                if 'unemployment' in state[iso]:
                    base_unemp = macro_future_preds.get('unemployment', {}).get((iso, year), np.nan)
                    if math.isnan(base_unemp): base_unemp = state[iso].get('unemployment', 0.0)
                    state[iso]['unemployment'] = base_unemp + 0.1 * (-net_mig / (new_pop+1e-9))
                if 'education' in state[iso]:
                    base_edu = macro_future_preds.get('education', {}).get((iso, year), np.nan)
                    if math.isnan(base_edu): base_edu = state[iso].get('education', 0.0)
                    state[iso]['education'] = base_edu
                if 'employment_sector' in state[iso]:
                    base_emp = macro_future_preds.get('employment_sector', {}).get((iso, year), np.nan)
                    if math.isnan(base_emp): base_emp = state[iso].get('employment_sector', 0.0)
                    state[iso]['employment_sector'] = base_emp

            # Save a compact outdf and keep net_by_origin mapping somewhere useful
            outdf = df_future[['orig','dest','predicted_mig_rate','predicted_mig_count'] + macro_delta_cols].copy()
            # add net_by_origin column to outdf for convenience (only for origins)
            outdf['_net_origin_map'] = None  # placeholder; we'll compute hover info separately
            all_predictions[key] = {'flows': outdf, 'net_by_origin': net_by_origin}
            elapsed = time.time() - t0
            safe_print(f"  -> Completed {key}. elapsed {elapsed:.1f}s")

    # cache pkl
    with open(QUARTERLY_PRED_PKL, "wb") as f:
        pickle.dump(all_predictions, f)
    safe_print("Saved all quarterly predictions to", QUARTERLY_PRED_PKL)

    # Also write CSV with flattened flows (option A — only create when generated)
    rows_for_csv = []
    for quarter_key, data in all_predictions.items():
        flows = data['flows']
        for rr in flows.itertuples(index=False):
            rows_for_csv.append({
                'quarter': quarter_key,
                'orig': rr.orig,
                'dest': rr.dest,
                'predicted_mig_rate': rr.predicted_mig_rate,
                'predicted_mig_count': rr.predicted_mig_count
            })
    csv_df = pd.DataFrame(rows_for_csv)
    csv_df.to_csv(QUARTERLY_PRED_CSV, index=False)
    safe_print("Saved flattened quarterly predictions CSV to", QUARTERLY_PRED_CSV)

# -------------------------
# 7) Aggregation for display helper
# -------------------------
def aggregate_for_display(pred_data):
    # pred_data is dict entry: {'flows': outdf, 'net_by_origin': {...}}
    df_future = pred_data['flows']
    net_map = pred_data['net_by_origin']
    # compute mean predicted_mig_rate per origin
    agg = df_future.groupby('orig')['predicted_mig_rate'].mean().reset_index()
    agg.columns = ['ISO3','predicted_mig_rate']
    if len(agg) == 0:
        return agg.set_index('ISO3'), {}
    # convert to percent for display (predicted_mig_rate is fraction)
    agg['predicted_mig_pct'] = agg['predicted_mig_rate'] * 100.0
    # Clip for display sanity
    agg['predicted_mig_pct'] = agg['predicted_mig_pct'].clip(lower=-100.0, upper=100.0)
    mn = agg['predicted_mig_pct'].min()
    mx = agg['predicted_mig_pct'].max()
    if math.isclose(mn, mx):
        agg['norm'] = 0.0
    else:
        agg['norm'] = (agg['predicted_mig_pct'] - mn) / (mx - mn)
    return agg.set_index('ISO3'), net_map

# -------------------------
# 8) Dash app & callbacks
# -------------------------
safe_print("Starting Dash app (UI)...")
app = Dash(__name__)
app.title = "Global Migration Predictor (Stacked, Dynamic)"

quarter_options = list(all_predictions.keys())

app.layout = html.Div([
    html.H2("Global Migration Predictor — Stacked Ensemble (Dynamic Macro↔Migration)", style={'textAlign':'center'}),
    html.Div("Slide through quarters 2026→2050. Hover a country to show top outgoing flows; click to lock/unlock arrows.", style={'textAlign':'center'}),
    dcc.Slider(
        id='quarter-slider',
        min=0, max=len(quarter_options)-1, value=0,
        marks={i: quarter_options[i] for i in range(0, len(quarter_options), max(1, len(quarter_options)//12))},
        step=1
    ),
    dcc.Store(id='clicked-countries', data=[]),
    dcc.Graph(id='migration-map', style={'height':'85vh'})
], style={'margin':'8px'})

@app.callback(
    Output('clicked-countries', 'data'),
    Input('migration-map', 'clickData'),
    State('clicked-countries', 'data')
)
def toggle_click(clickData, clicked):
    if clickData is None:
        return clicked
    try:
        pt = clickData['points'][0]
        country_iso = pt.get('location') or pt.get('text')
        if not country_iso:
            return clicked
        country_iso = str(country_iso).strip()
        if country_iso in clicked:
            clicked.remove(country_iso)
        else:
            clicked.append(country_iso)
        return clicked
    except Exception:
        return clicked

def find_country_coords(code_or_name):
    if pd.isna(code_or_name) or code_or_name is None:
        return None
    c = str(code_or_name).strip()
    # direct ISO3
    if c in country_coords:
        return country_coords[c]
    # try ISO2 -> ISO3
    try:
        iso3 = iso2_to_iso3(c)
        if iso3 and iso3 in country_coords:
            return country_coords[iso3]
    except Exception:
        pass
    # match centroids by ISO3 or COUNTRY
    matches = centroids[centroids['ISO3']==c]
    if len(matches)>0:
        r = matches.iloc[0]
        return (float(r['latitude']), float(r['longitude']))
    matches = centroids[centroids.apply(lambda r: str(r.get('COUNTRY','')).strip().upper()==c.upper(), axis=1)]
    if len(matches)>0:
        r = matches.iloc[0]
        return (float(r['latitude']), float(r['longitude']))
    return None

@app.callback(
    Output('migration-map', 'figure'),
    Input('quarter-slider', 'value'),
    Input('migration-map', 'hoverData'),
    State('clicked-countries', 'data')
)
def update_map(slider_index, hoverData, clicked_countries):
    quarter_key = quarter_options[slider_index]
    pred_entry = all_predictions[quarter_key]
    df_future = pred_entry['flows']
    agg, net_map = aggregate_for_display(pred_entry)
    merged = centroids.set_index('ISO3').join(agg[['predicted_mig_pct','norm']], how='left').reset_index()
    locations = merged['ISO3'].tolist()
    zvals = merged['predicted_mig_pct'].fillna(0).tolist()
    country_names = merged['COUNTRY'].tolist()

    fig = go.Figure()
    # Choropleth with custom two-color gradient (low -> #52B4FF ; high -> #00233D)
    colorscale_custom = [[0, '#52B4FF'], [1, '#00233D']]
    # We'll build a hovertemplate that only shows Country, Net migration count, Migration change %
    # Build net_map for all origins (if missing default 0)
    hover_texts = []
    for iso, country in zip(merged['ISO3'], country_names):
        net = net_map.get(iso, 0.0)
        # try to format net as integer with commas
        try:
            net_str = f"{int(round(net)):,}"
        except Exception:
            net_str = "0"
        pct = merged.loc[merged['ISO3']==iso, 'predicted_mig_pct'].iloc[0] if iso in merged['ISO3'].values else 0.0
        pct = float(np.clip(pct, -100.0, 100.0))
        hover_texts.append(f"{country}<br>Net migration: {net_str}<br>Change: {pct:.2f}%")
    fig.add_trace(go.Choropleth(
        locations=locations,
        z=zvals,
        text=hover_texts,  # used in hovertemplate as %{text}
        colorscale=colorscale_custom,
        colorbar_title='Predicted outflow (%)',
        marker_line_color='black',
        locationmode='ISO-3',
        hovertemplate="%{text}<extra></extra>"
    ))

    # determine active origins: clicked + hovered
    active = set(clicked_countries or [])
    if hoverData and 'points' in hoverData and len(hoverData['points'])>0:
        try:
            hp = hoverData['points'][0]
            hover_iso = hp.get('location') or None
            if hover_iso:
                active.add(str(hover_iso))
        except Exception:
            pass

    # add arrows for each active origin
    for origin, group in df_future.groupby('orig'):
        if origin not in active:
            continue
        origin_coords = find_country_coords(origin)
        if not origin_coords:
            continue
        lat0, lon0 = origin_coords
        # pick top destinations by predicted_mig_count (descending)
        sorted_group = group.sort_values('predicted_mig_count', ascending=False).reset_index(drop=True)
        selected_dests = []
        # select from sorted_group first
        for _, row in sorted_group.iterrows():
            if len(selected_dests) >= TOP_K_ARROWS:
                break
            dest = row['dest']
            if dest not in selected_dests:
                selected_dests.append(dest)
        # if not enough, append fallback list in order
        for fb in FALLBACK_ISO3:
            if len(selected_dests) >= TOP_K_ARROWS:
                break
            if fb not in selected_dests:
                selected_dests.append(fb)
        # draw arrows for selected_dests in the same order
        # compute max_count for linewidth scaling
        max_count = sorted_group['predicted_mig_count'].max() if len(sorted_group)>0 else 1.0
        # compute norm for opacity
        origin_norm = float(agg['norm'].get(origin, 0.0)) if origin in agg.index else 0.0
        color_alpha = 0.25 + 0.6 * origin_norm
        for dest in selected_dests[:TOP_K_ARROWS]:
            dest_coords = find_country_coords(dest)
            if dest_coords is None:
                continue
            lat1, lon1 = dest_coords
            # compute migrants & width from group if available else small default
            rowmatch = sorted_group[sorted_group['dest']==dest]
            migrants = int(rowmatch['predicted_mig_count'].iloc[0]) if len(rowmatch)>0 else 0
            rel = (rowmatch['predicted_mig_count'].iloc[0] / max_count) if (len(rowmatch)>0 and max_count>0) else 0.0
            width = 0.6 + 3.4 * rel
            # Draw line with no hoverinfo (arrows/tooltips removed per request)
            fig.add_trace(go.Scattergeo(
                lon=[lon0, lon1],
                lat=[lat0, lat1],
                mode='lines',
                line=dict(width=width, color=f"rgba(0,0,255,{color_alpha:.3f})"),
                hoverinfo='none',
                showlegend=False
            ))

    fig.update_geos(showcoastlines=True, showcountries=True, showframe=False)
    fig.update_layout(title=f"Predicted Migration — {quarter_key}",
                      geo=dict(projection_type='natural earth'),
                      margin=dict(l=0,r=0,t=40,b=0))
    return fig

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    safe_print("App ready. Launching Dash (http://127.0.0.1:8050)...")
    try:
        app.run_server(debug=True, port=8050)
    except Exception:
        app.run(debug=True)
