"""
CropCast training pipeline.

1. Audits the "99% R²" setup (random split on the raw, duplicated data).
2. Deduplicates to one row per (country, crop, year).
3. Trains a LightGBM model that predicts the *change* in log-yield vs last year.
4. Evaluates on unseen future years (2009-2013) against honest baselines.
5. Calibrates 80% prediction intervals with split-conformal on 2006-2008.
6. Refits on all years and writes artifacts for the API.

Run:  python train.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "yield_df.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

KEY = ["Area", "Item", "Year"]
FEATURES = ["Area", "Item", "Year", "rain", "pest", "temp",
            "trend", "temp_d", "pest_d", "lag1_vs_roll"]
PARAMS = dict(objective="huber", alpha=0.15, learning_rate=0.02, n_estimators=600,
              num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
INTERVAL = 0.80


def audit_original(raw: pd.DataFrame) -> dict:
    """Reproduce the Extra Trees + random split result and test it on harder splits."""
    X = pd.get_dummies(raw[["Area", "Item", "Year", "average_rain_fall_mm_per_year",
                            "pesticides_tonnes", "avg_temp"]], columns=["Area", "Item"], dtype=float)
    y = raw["hg/ha_yield"]
    out = {}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    p = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(Xtr, ytr).predict(Xte)
    keys = raw[KEY].astype(str).agg("|".join, axis=1)
    out["random_split"] = dict(r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                               leaked_share=float(keys.loc[Xte.index].isin(set(keys.loc[Xtr.index])).mean()))

    tr = raw.Year <= 2008
    p = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(X[tr], y[tr]).predict(X[~tr])
    out["future_years"] = dict(r2=r2_score(y[~tr], p), mae=mean_absolute_error(y[~tr], p))

    r2s, maes = [], []
    for a, b in GroupKFold(5).split(X, y, raw.Area):
        p = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=0).fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b])
        r2s.append(r2_score(y.iloc[b], p)); maes.append(mean_absolute_error(y.iloc[b], p))
    out["unseen_countries"] = dict(r2=float(np.mean(r2s)), mae=float(np.mean(maes)))
    return out


def build_table(raw: pd.DataFrame) -> pd.DataFrame:
    d = (raw.groupby(KEY, as_index=False)
            .agg(y=("hg/ha_yield", "first"), rain=("average_rain_fall_mm_per_year", "first"),
                 pest=("pesticides_tonnes", "first"), temp=("avg_temp", "mean"))
            .sort_values(KEY).reset_index(drop=True))
    return d


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(KEY).copy()
    g = d.groupby(["Area", "Item"], observed=True)
    d["lag1"] = g.y.shift(1)
    d["lag2"] = g.y.shift(2)
    d["roll3"] = g.y.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    d["trend"] = np.log1p(d.lag1) - np.log1p(d.lag2)
    d["temp_d"] = d.temp - g.temp.shift(1)
    d["pest_d"] = np.log1p(d.pest) - np.log1p(g.pest.shift(1))
    d["lag1_vs_roll"] = np.log1p(d.lag1) - np.log1p(d.roll3)
    d["target"] = np.log1p(d.y) - np.log1p(d.lag1)
    return d


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(mae=float(mean_absolute_error(y, p)), r2=float(r2_score(y, p)),
                median_pct_error=float(np.median(np.abs(p - y) / y)))


def main():
    raw = pd.read_csv(DATA).drop(columns="Unnamed: 0")
    print("Auditing original setup…")
    audit = audit_original(raw)

    table = build_table(raw)
    cats = {c: sorted(table[c].unique()) for c in ["Area", "Item"]}
    d = add_features(table)
    for c in ["Area", "Item"]:
        d[c] = pd.Categorical(d[c], categories=cats[c])
    d = d[d.lag1.notna()].reset_index(drop=True)

    tr, cal, te = d.Year <= 2005, (d.Year > 2005) & (d.Year <= 2008), d.Year > 2008

    # Conformal: fit on <=2005, calibrate residuals on 2006-08
    m_cal = lgb.LGBMRegressor(**PARAMS).fit(d.loc[tr, FEATURES], d.loc[tr, "target"])
    res = np.abs(d.loc[cal, "target"] - m_cal.predict(d.loc[cal, FEATURES]))
    q = float(np.quantile(res, INTERVAL))

    # Evaluate on 2009-13 with a model fit on <=2008
    m_eval = lgb.LGBMRegressor(**PARAMS).fit(d.loc[tr | cal, FEATURES], d.loc[tr | cal, "target"])
    base = np.log1p(d.loc[te, "lag1"])
    step = m_eval.predict(d.loc[te, FEATURES])
    pred = np.expm1(base + step)
    lo, hi = np.expm1(base + step - q), np.expm1(base + step + q)
    yt = d.loc[te, "y"]
    evaluation = dict(
        cropcast=metrics(yt, pred),
        naive_last_year=metrics(yt, d.loc[te, "lag1"]),
        rolling_3yr_mean=metrics(yt, d.loc[te, "roll3"]),
        interval_coverage=float(((yt >= lo) & (yt <= hi)).mean()),
        n_test=int(te.sum()),
    )

    # Final model on everything
    final = lgb.LGBMRegressor(**PARAMS).fit(d[FEATURES], d["target"])
    final.booster_.save_model(str(ART / "model.txt"))
    gain = final.booster_.feature_importance("gain")
    importance = {f: float(v) for f, v in zip(FEATURES, gain / gain.sum())}

    history = {}
    for (area, item), grp in table.groupby(["Area", "Item"]):
        history.setdefault(area, {})[item] = grp[["Year", "y", "rain", "pest", "temp"]].round(3).to_dict("list")

    meta = dict(features=FEATURES, categories=cats, conformal_q=q, interval=INTERVAL,
                last_year=int(table.Year.max()), audit=audit, evaluation=evaluation,
                importance=importance, params=PARAMS)
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))
    (ART / "history.json").write_text(json.dumps(history))

    print(json.dumps(dict(audit=audit, evaluation=evaluation), indent=2))


if __name__ == "__main__":
    main()
