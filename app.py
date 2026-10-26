"""CropCast API — FastAPI service around the LightGBM yield-change model."""
import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

booster = lgb.Booster(model_file=str(ART / "model.txt"))
META = json.loads((ART / "meta.json").read_text())
HISTORY = json.loads((ART / "history.json").read_text())
FEATURES = META["features"]
CATS = META["categories"]
Q = META["conformal_q"]

GROUPS = {
    "Recent momentum": ["trend", "lag1_vs_roll"],
    "Temperature": ["temp", "temp_d"],
    "Rainfall": ["rain"],
    "Pesticide use": ["pest", "pest_d"],
    "Country & crop": ["Area", "Item"],
    "Long-run year trend": ["Year"],
}

app = FastAPI(title="CropCast", version="1.0",
              description="Next-season crop yield forecasts with calibrated intervals.")


class Scenario(BaseModel):
    area: str
    item: Optional[str] = None
    rain: Optional[float] = Field(None, ge=0, le=5000, description="Annual rainfall, mm")
    temp: Optional[float] = Field(None, ge=-10, le=45, description="Average temperature, °C")
    pest: Optional[float] = Field(None, ge=0, description="Pesticide use, tonnes")


def _series(area: str, item: str) -> dict:
    try:
        return HISTORY[area][item]
    except KeyError:
        raise HTTPException(404, f"No history for {item} in {area}.")


def _rows(area: str, items: list[str], rain, temp, pest):
    rows, ctx = [], []
    for item in items:
        h = _series(area, item)
        y, yrs = h["y"], h["Year"]
        if len(y) < 2:
            continue
        r = rain if rain is not None else h["rain"][-1]
        t = temp if temp is not None else h["temp"][-1]
        p = pest if pest is not None else h["pest"][-1]
        roll3 = float(np.mean(y[-3:]))
        rows.append(dict(
            Area=area, Item=item, Year=yrs[-1] + 1, rain=r, pest=p, temp=t,
            trend=np.log1p(y[-1]) - np.log1p(y[-2]),
            temp_d=t - h["temp"][-1],
            pest_d=np.log1p(p) - np.log1p(h["pest"][-1]),
            lag1_vs_roll=np.log1p(y[-1]) - np.log1p(roll3),
        ))
        ctx.append(dict(item=item, last=y[-1], last_year=yrs[-1], inputs=dict(rain=r, temp=t, pest=p)))
    if not rows:
        raise HTTPException(422, "Not enough history to forecast (needs two seasons).")
    X = pd.DataFrame(rows)[FEATURES]
    for c in ["Area", "Item"]:
        X[c] = pd.Categorical(X[c], categories=CATS[c])
    return X, ctx


def _forecast(X, ctx):
    step = booster.predict(X)
    out = []
    for s, c in zip(step, ctx):
        base = np.log1p(c["last"])
        out.append(dict(
            item=c["item"], year=c["last_year"] + 1, inputs=c["inputs"],
            yield_hg_ha=float(np.expm1(base + s)),
            low=float(np.expm1(base + s - Q)), high=float(np.expm1(base + s + Q)),
            last_yield=c["last"], change_pct=float(np.expm1(s) * 100),
        ))
    return out


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    return dict(
        countries={a: sorted(v.keys()) for a, v in sorted(HISTORY.items())},
        last_year=META["last_year"], interval=META["interval"],
        evaluation=META["evaluation"], audit=META["audit"],
    )


@app.get("/api/history")
def history(area: str, item: str):
    return _series(area, item)


@app.post("/api/predict")
def predict(s: Scenario):
    if not s.item:
        raise HTTPException(422, "Choose a crop.")
    X, ctx = _rows(s.area, [s.item], s.rain, s.temp, s.pest)
    res = _forecast(X, ctx)[0]
    contrib = booster.predict(X, pred_contrib=True)[0]
    by_feat = dict(zip(FEATURES, contrib[:-1]))
    res["drivers"] = sorted(
        [dict(name=g, pct=float(np.expm1(sum(by_feat[f] for f in fs)) * 100)) for g, fs in GROUPS.items()],
        key=lambda d: -abs(d["pct"]))
    res["history"] = _series(s.area, s.item)
    return res


@app.post("/api/compare")
def compare(s: Scenario):
    """Forecast every crop grown in the country under one climate scenario."""
    X, ctx = _rows(s.area, list(HISTORY.get(s.area, {}).keys()), s.rain, s.temp, s.pest)
    return sorted(_forecast(X, ctx), key=lambda r: -r["change_pct"])


@app.post("/api/sensitivity")
def sensitivity(s: Scenario, variable: str = "temp", points: int = 25):
    if variable not in {"temp", "rain", "pest"}:
        raise HTTPException(422, "variable must be temp, rain or pest.")
    h = _series(s.area, s.item)
    centre = h[variable][-1]
    span = {"temp": (centre - 4, centre + 4), "rain": (centre * 0.5, centre * 1.5),
            "pest": (centre * 0.25, centre * 2.0)}[variable]
    grid = np.linspace(*span, points)
    out = []
    for v in grid:
        kw = dict(rain=s.rain, temp=s.temp, pest=s.pest)
        kw[variable] = float(v)
        X, ctx = _rows(s.area, [s.item], kw["rain"], kw["temp"], kw["pest"])
        out.append(dict(value=float(v), yield_hg_ha=_forecast(X, ctx)[0]["yield_hg_ha"]))
    return out


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
