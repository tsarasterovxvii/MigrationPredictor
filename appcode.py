# migration_app_final_stack_dynamic_fixed_gui.py
# Full integrated stacked migration predictor + dynamic macro↔migration coupling + Dash UI
# Save and run: python migration_app_final_stack_dynamic_fixed_gui.py
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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor

import xgboost as xgb

# optional LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

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
MODEL_META_FILE = CACHE_DIR / "stacked_model_meta.pkl"   # metadata (feature names)
MACRO_MODELS_FILE = CACHE_DIR / "macro_models.pkl"
MACRO_FUTURE_FILE = CACHE_DIR / "macro_future_preds.pkl"
QUARTERLY_PRED_FILE = CACHE_DIR / "predictions_quarterly_2026_2050.pkl"

# Simulation horizon
START_YEAR = 2026
END_YEAR = 2050
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

# UI / behavior
TOP_K_ARROWS = 5

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

# map ISO3 -> (lat, lon)
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
                X = grp[['year']].values
                y = grp['value'].values
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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
# 4) Prepare migration model features & training dataset
# -------------------------
safe_print("Preparing migration model features...")
base_candidates = ['year0','sd_drop_neg','sd_rev_neg','da_min_open','da_min_closed','da_pb_closed']
base_feats = [c for c in base_candidates if c in bilat.columns]
safe_print("Base features used:", base_feats)

# categorical codes for origin/destination
bilat['orig_code'] = bilat['orig'].astype('category').cat.codes
bilat['dest_code'] = bilat['dest'].astype('category').cat.codes

# start with base features + codes
feature_cols = base_feats + ['orig_code','dest_code']

# IMPORTANT FIX:
# Add seasonal placeholder columns 'quarter_sin' and 'quarter_cos' to training schema so later dynamic predictions
# which include these columns do not cause a feature-name mismatch. Historical rows use neutral values (0,1).
if 'quarter_sin' not in bilat.columns:
    bilat['quarter_sin'] = 0.0   # neutral for historic rows
if 'quarter_cos' not in bilat.columns:
    bilat['quarter_cos'] = 1.0
# ensure these are included in training feature list
if 'quarter_sin' not in feature_cols:
    feature_cols.append('quarter_sin')
if 'quarter_cos' not in feature_cols:
    feature_cols.append('quarter_cos')

# attach historical macro value columns (for training) if possible — use macro_dfs historical values where year0 exists
for mname, df_macro in macro_dfs.items():
    colname = f"{mname}_orig_hist"
    lookup = {(r.ISO3, int(r.year)): r.value for r in df_macro.itertuples(index=False)}
    def macro_for_row(row):
        iso = row['orig']
        # prefer row's year0 if available
        year = int(row['year0']) if 'year0' in row and not pd.isna(row['year0']) else None
        if year is not None and (iso, year) in lookup:
            return lookup[(iso, year)]
        # fallback: last observed
        sub = df_macro[df_macro['ISO3']==iso].sort_values('year', ascending=False)
        if len(sub)>0:
            return float(sub.iloc[0]['value'])
        return np.nan
    try:
        bilat[colname] = bilat.apply(macro_for_row, axis=1)
        feature_cols.append(colname)
        safe_print(f"Attached macro historical feature: {colname}")
    except Exception as e:
        safe_print(f"Could not attach {mname} historical as feature: {e}")

# make feature_cols unique and preserve order
feature_cols = list(dict.fromkeys(feature_cols))
safe_print("Final feature columns (for training):", feature_cols)

# -------------------------
# 5) Build stacked ensemble and train (or load cached), with feature-name metadata handling
# -------------------------
def load_model_and_meta():
    """Load cached model and its metadata (feature names) if both exist, else return (None, None)."""
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

def save_model_and_meta(mdl, feature_list):
    with open(STACKED_MODEL_FILE, "wb") as f:
        pickle.dump(mdl, f)
    # ensure feature_list is unique (no duplicates) and is a plain list
    feature_list_unique = list(dict.fromkeys(list(feature_list)))
    meta = {'feature_cols': feature_list_unique}
    with open(MODEL_META_FILE, "wb") as f:
        pickle.dump(meta, f)

# Attempt to load cached model + meta
stack_model, model_meta = load_model_and_meta()

# if model exists but meta mismatch, force retrain
need_retrain = False
if stack_model is not None and model_meta is not None:
    trained_cols = model_meta.get('feature_cols', [])
    set_trained = set([str(c) for c in trained_cols])
    set_current = set([str(c) for c in feature_cols])
    if set_trained != set_current:
        safe_print("Cached stacked model feature set differs from current feature set. Will retrain model to include the new features (or remove extras).")
        safe_print("Trained cols (cached):", trained_cols)
        safe_print("Current cols  (wanted):", feature_cols)
        need_retrain = True
else:
    need_retrain = True

if need_retrain:
    safe_print("Building stacked ensemble (RF, XGB, HistGB, optional LGB) and training — this may be slow...")
    estimators = []
    rf_pipe = make_pipeline(SimpleImputer(strategy='mean'), RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    estimators.append(('rf', rf_pipe))
    xgb_pipe = make_pipeline(SimpleImputer(strategy='mean'), xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0))
    estimators.append(('xgb', xgb_pipe))
    hgb_pipe = make_pipeline(SimpleImputer(strategy='mean'), HistGradientBoostingRegressor(max_iter=200, random_state=42))
    estimators.append(('hgb', hgb_pipe))
    if HAS_LGB:
        lgb_pipe = make_pipeline(SimpleImputer(strategy='mean'), lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42))
        estimators.append(('lgb', lgb_pipe))
        safe_print("LightGBM included in stack.")
    else:
        safe_print("LightGBM not available; skipped.")

    final_est = make_pipeline(SimpleImputer(strategy='mean'), xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0))
    stack_model = StackingRegressor(estimators=estimators, final_estimator=final_est, cv=STACK_CV, n_jobs=-1, passthrough=False)

    X = bilat[feature_cols].fillna(0)
    y = bilat['mig_rate'].fillna(0)
    safe_print("Fitting stacked model on", X.shape[0], "rows and", X.shape[1], "features ...")
    stack_model.fit(X, y)
    # Save model and the exact feature order (unique)
    save_model_and_meta(stack_model, X.columns.tolist())
    safe_print("Saved stacked model to", STACKED_MODEL_FILE)
else:
    safe_print("Loaded cached stacked migration model and metadata.")

# -------------------------
# 6) Coupled dynamic simulation across quarters (Q1 2026 — Q4 2050)
# -------------------------
expected_keys = [f"{q} {y}" for y in range(START_YEAR, END_YEAR+1) for q in QUARTERS]
regen = False
if QUARTERLY_PRED_FILE.exists():
    safe_print("Found cached quarterly predictions; loading...")
    with open(QUARTERLY_PRED_FILE, "rb") as f:
        all_predictions = pickle.load(f)
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
    # Helper to get baseline macro for iso/year
    def macro_baseline(mname, iso, year):
        val = macro_future_preds.get(mname, {}).get((iso, year), np.nan) if isinstance(macro_future_preds, dict) else np.nan
        return val

    # initialize using last observed historical value if available, else baseline for START_YEAR
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

    # We'll iterate year by year, quarter by quarter
    total_steps = (END_YEAR - START_YEAR + 1) * 4
    step_idx = 0
    t0 = time.time()
    # Precompute row-level static features that never change per-row except year0/quarter and macro columns
    static_columns = ['orig','dest','orig_code','dest_code']
    for col in static_columns:
        if col not in bilat.columns:
            bilat[col] = ""
    bilat_static = bilat[static_columns].copy()

    # get trained feature order from meta (if exists) so we can reindex prediction X to that order before predict
    trained_feature_order = None
    if MODEL_META_FILE.exists():
        try:
            with open(MODEL_META_FILE, "rb") as f:
                meta = pickle.load(f)
            trained_feature_order = meta.get('feature_cols', None)
            # ensure trained_feature_order is unique and in stable order
            if trained_feature_order is not None:
                trained_feature_order = list(dict.fromkeys(trained_feature_order))
        except Exception:
            trained_feature_order = None

    for year in range(START_YEAR, END_YEAR+1):
        for qidx, quarter in enumerate(QUARTERS, start=1):
            step_idx += 1
            key = f"{quarter} {year}"
            safe_print(f"Simulating {key} ({step_idx}/{total_steps}) ...")
            # Build df_future from static template
            df_future = bilat_static.copy()
            # add base numeric features if present in original bilat
            for feat in base_feats:
                if feat in bilat.columns:
                    df_future[feat] = bilat[feat]
                else:
                    df_future[feat] = 0.0
            # year/quarter features
            df_future['year0'] = year
            df_future['quarter'] = qidx
            df_future['quarter_sin'] = math.sin(2*math.pi*(qidx/4.0))
            df_future['quarter_cos'] = math.cos(2*math.pi*(qidx/4.0))
            # attach per-origin macro current state for this year (we'll use state[iso] values)
            # Also add baseline macro forecasts for that year from macro_future_preds to use as baseline and for adjustment
            for mname in macro_future_preds.keys():
                col_future = f"{mname}_orig_future"
                col_baseline = f"{mname}_orig_baseline"
                vals_f = []
                vals_b = []
                for iso in df_future['orig'].values:
                    iso = str(iso)
                    # baseline prediction (from RF macro model previously saved)
                    baseline_val = macro_future_preds.get(mname, {}).get((iso, year), np.nan)
                    vals_b.append(baseline_val)
                    # current dynamic state value (if present) — use state (e.g., population, gdp) else baseline
                    s = state.get(iso, {})
                    v = s.get(mname) if mname in s else None
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        v = baseline_val
                    vals_f.append(v)
                df_future[col_future] = vals_f
                df_future[col_baseline] = vals_b

            # Build Xf with feature_cols plus macro future columns + quarter sin/cos
            macro_cols_this = [c for c in df_future.columns if c.endswith('_orig_future')]
            # Combine desired columns but remove duplicates while preserving order
            combined_cols = [c for c in feature_cols if c in df_future.columns] + macro_cols_this + ['quarter_sin','quarter_cos']
            Xf_cols = list(dict.fromkeys(combined_cols))  # dedupe preserving order

            # ensure all columns exist
            for c in Xf_cols:
                if c not in df_future.columns:
                    df_future[c] = 0.0
            Xf = df_future[Xf_cols].fillna(0)

            # Important: reindex Xf to the exact column order and names used during training,
            # so the pipelines and imputers inside stacked model receive identical feature names.
            if trained_feature_order is not None:
                # ensure all trained feature cols exist in Xf; if not, create with 0
                missing_in_Xf = [c for c in trained_feature_order if c not in Xf.columns]
                for c in missing_in_Xf:
                    Xf[c] = 0.0
                # make a deduped order from trained_feature_order
                trained_order_dedup = list(dict.fromkeys(trained_feature_order))
                # Reindex safely (this will raise if something else wrong, so we catch)
                try:
                    Xf = Xf.reindex(columns=trained_order_dedup, fill_value=0.0)
                except ValueError as e:
                    # fallback: if reindex fails due to duplicates (shouldn't after dedupe) or other reason,
                    # fall back to selecting the intersection preserving trained order
                    safe_print("Warning: reindex failed; falling back to best-effort column ordering:", e)
                    cols_int = [c for c in trained_order_dedup if c in Xf.columns]
                    Xf = Xf[cols_int].copy()
            else:
                # if meta missing, ensure Xf columns are unique (they are) - proceed
                pass

            # Predict migration rates
            preds = stack_model.predict(Xf)
            # season adj (tiny)
            season_adj = 1.0 + 0.02 * df_future['quarter_sin']
            preds = preds * season_adj
            df_future['predicted_mig_rate'] = preds
            # Convert to predicted counts using current origin population state
            counts = []
            for iso, rate in zip(df_future['orig'].values, df_future['predicted_mig_rate'].values):
                iso = str(iso)
                pop = state.get(iso, {}).get('population', 1_000_000.0)
                # decide scale: if model outputs fraction (<1.5) treat as fraction; else percent
                if abs(rate) <= 1.5:
                    cnt = rate * pop
                else:
                    cnt = (rate / 100.0) * pop
                counts.append(max(0.0, cnt))
            df_future['predicted_mig_count'] = counts
            # Aggregate flows per origin to compute net change in population (outflows reduce origin population; inflows increase dest)
            outflow = df_future.groupby('orig')['predicted_mig_count'].sum().to_dict()
            inflow = df_future.groupby('dest')['predicted_mig_count'].sum().to_dict()
            # Update dynamic state: population, then derive gdp_percap and adjust GDP baseline with some sensitivity
            POP_TO_GDP_ELASTICITY = 0.2   # how population change proportionally affects GDP baseline
            MIGRATION_TO_POP_SCALER = 1.0 # direct addition/subtraction scaling
            for iso in list(state.keys()):
                iso = str(iso)
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

            # Save outdf compact form (ensure macro_cols_this defined)
            outdf = df_future[['orig','dest','predicted_mig_rate','predicted_mig_count'] + macro_cols_this].copy()
            all_predictions[key] = outdf
            elapsed = time.time() - t0
            safe_print(f"  -> Completed {key}. elapsed {elapsed:.1f}s")
    # cache
    with open(QUARTERLY_PRED_FILE, "wb") as f:
        pickle.dump(all_predictions, f)
    safe_print("Saved all quarterly predictions to", QUARTERLY_PRED_FILE)

# -------------------------
# 7) Aggregation for display helper
# -------------------------
def aggregate_for_display(df_future):
    agg = df_future.groupby('orig')['predicted_mig_rate'].mean().reset_index()
    agg.columns = ['ISO3','predicted_mig_rate']
    if len(agg) == 0:
        return agg.set_index('ISO3')
    if agg['predicted_mig_rate'].abs().max() <= 1.5:
        agg['predicted_mig_pct'] = agg['predicted_mig_rate'] * 100.0
    else:
        agg['predicted_mig_pct'] = agg['predicted_mig_rate']
    mn = agg['predicted_mig_pct'].min()
    mx = agg['predicted_mig_pct'].max()
    if math.isclose(mn, mx):
        agg['norm'] = 0.0
    else:
        agg['norm'] = (agg['predicted_mig_pct'] - mn) / (mx - mn)
    return agg.set_index('ISO3')

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

@app.callback(
    Output('migration-map', 'figure'),
    Input('quarter-slider', 'value'),
    Input('migration-map', 'hoverData'),
    State('clicked-countries', 'data')
)
def update_map(slider_index, hoverData, clicked_countries):
    quarter_key = quarter_options[slider_index]
    df_future = all_predictions[quarter_key]
    agg = aggregate_for_display(df_future)
    merged = centroids.set_index('ISO3').join(agg[['predicted_mig_pct','norm']], how='left').reset_index()
    locations = merged['ISO3'].tolist()
    zvals = merged['predicted_mig_pct'].fillna(0).tolist()
    textvals = merged['COUNTRY'].tolist()

    fig = go.Figure()
    # Choropleth — color by predicted outflow percent
    fig.add_trace(go.Choropleth(
        locations=locations,
        z=zvals,
        text=textvals,
        colorscale='Blues',
        colorbar_title='Predicted outflow (%)',
        marker_line_color='black',
        locationmode='ISO-3',
        hovertemplate="%{text}<br>Outflow: %{z:.3f}%<extra></extra>"
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
        if origin not in country_coords:
            continue
        lat0, lon0 = country_coords[origin]
        normv = float(agg['norm'].get(origin, 0.0)) if origin in agg.index else 0.0
        color_alpha = 0.25 + 0.6*normv
        topg = group.sort_values('predicted_mig_rate', ascending=False).head(TOP_K_ARROWS)
        for _, row in topg.iterrows():
            dest = row['dest']
            if dest not in country_coords:
                continue
            lat1, lon1 = country_coords[dest]
            rate = row['predicted_mig_rate']
            if abs(rate) <= 1.5:
                display_rate = rate * 100.0
            else:
                display_rate = rate
            migrants = int(max(0.0, row.get('predicted_mig_count', 0)))
            hover_text = f"{origin} → {dest}<br>Rate: {display_rate:.3f}%<br>Migrants (est): {migrants:,}"
            fig.add_trace(go.Scattergeo(
                lon=[lon0, lon1],
                lat=[lat0, lat1],
                mode='lines',
                line=dict(width=0.6, color=f"rgba(0,0,255,{color_alpha:.3f})"),
                hoverinfo='text',
                text=hover_text,
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
