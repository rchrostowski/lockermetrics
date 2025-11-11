import re
import pandas as pd
import numpy as np
from dateutil import parser as dparse
import duckdb
import plotly.express as px

SUPPORTED_METRICS = {
    "readiness": "readiness",
    "acwr": "acwr",
    "load": "load",
    "sleep": "sleep_hours",
    "hrv": "hrv_ms",
    "rhr": "rhr_bpm",
    "strain": "strain_0_21",
    "soreness": "soreness_1_5"
}
SUPPORTED_CHARTS = ("line","bar","scatter","heatmap","table")

def _last_n_days(df, n):
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=n)
    return df[df["date"] >= cutoff]

def _between_dates(df, start, end):
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    s = dparse.parse(start).date(); e = dparse.parse(end).date()
    return df[(df["date"] >= pd.to_datetime(s)) & (df["date"] <= pd.to_datetime(e))]

def _pick_metric(text):
    for k, col in SUPPORTED_METRICS.items():
        if re.search(rf"\b{k}\b", text, re.I): return col
    return "readiness"

def _pick_chart(text, default="line"):
    for c in SUPPORTED_CHARTS:
        if re.search(rf"\b{c}\b", text, re.I): return c
    if "vs" in text.lower() or "correl" in text.lower(): return "scatter"
    if "top" in text.lower() or "by " in text.lower():  return "bar"
    return default

def _extract_player(text, df):
    names = df["display_name"].dropna().unique().tolist()
    for n in sorted(names, key=len, reverse=True):
        if re.search(rf"\b{re.escape(n)}\b", text, re.I): return n
    return None

def _extract_team(text, df):
    teams = df["team"].dropna().unique().tolist()
    for t in sorted(teams, key=len, reverse=True):
        if re.search(rf"\b{re.escape(t)}\b", text, re.I): return t
    return None

def _extract_top_n(text, default=10):
    m = re.search(r"\btop\s+(\d+)", text, re.I)
    if m:
        try: return int(m.group(1))
        except: pass
    return default

def _maybe_sql(text, df):
    if not text.strip().lower().startswith("sql:"): return None
    q = text.strip()[4:].strip()
    con = duckdb.connect()
    con.register("metrics", df)
    try:
        out = con.execute(q).fetch_df()
        return out
    except Exception as e:
        return pd.DataFrame({"error":[str(e)]})

def interpret_question(text, df):
    # Optional SQL fast-path
    sql_out = _maybe_sql(text, df)
    if isinstance(sql_out, pd.DataFrame):
        return (None, sql_out, "SQL result")

    t = text.strip()
    metric = _pick_metric(t)
    chart  = _pick_chart(t)
    n      = _extract_top_n(t, default=10)
    player = _extract_player(t, df)
    team   = _extract_team(t, df)

    work = df.copy()

    # Entities
    if team:
        work = work[work["team"].str.contains(team, case=False, na=False)]
    if player:
        work = work[work["display_name"].str.contains(player, case=False, na=False)]

    # Time
    last_n = None
    m = re.search(r"last\s+(\d+)\s*day", t, re.I)
    if m: last_n = int(m.group(1))
    rng = re.search(r"from\s+([0-9/\-\.]+)\s+(to|through)\s+([0-9/\-\.]+)", t, re.I)

    if last_n: work = _last_n_days(work, last_n)
    elif rng:  work = _between_dates(work, rng.group(1), rng.group(3))

    # Red flags table
    if re.search(r"\b(red flags|at risk|injur)\b", t, re.I):
        latest = pd.to_datetime(work["date"]).max()
        today  = work[pd.to_datetime(work["date"])==latest].copy()
        if "flag" in today.columns:
            today = today.sort_values(["flag","readiness","acwr"], ascending=[True, True, False])
        return (None, today[["date","display_name","team","position","readiness","acwr","flag"]], "Risk table")

    # X vs Y scatter
    vs = re.search(r"(readiness|acwr|load|sleep|hrv|rhr|strain|soreness)\s*(?:vs|against)\s*(readiness|acwr|load|sleep|hrv|rhr|strain|soreness)", t, re.I)
    if vs:
        x = SUPPORTED_METRICS[vs.group(1).lower()]
        y = SUPPORTED_METRICS[vs.group(2).lower()]
        fig = px.scatter(work, x=x, y=y, color="display_name", hover_data=["date","team"])
        return (fig, work[["date","display_name","team",x,y]], f"Scatter: {x} vs {y}")

    # Top-N by metric (latest day)
    if re.search(r"\btop\b", t, re.I):
        latest = pd.to_datetime(work["date"]).max()
        today  = work[pd.to_datetime(work["date"])==latest].copy()
        top = today.sort_values(metric, ascending=False).head(n)
        if chart == "bar":
            fig = px.bar(top, x="display_name", y=metric, color="team")
            return (fig, top[["display_name","team","position",metric]], f"Top {n} by {metric} (today)")
        return (None, top[["display_name","team","position",metric]], f"Top {n} by {metric} (today)")

    # Per-player time series (default line)
    if player or chart == "line":
        ts = (work.groupby("date")[metric].mean().reset_index().sort_values("date"))
        fig = px.line(ts, x="date", y=metric, markers=True)
        label = f"{metric} over time" + (f" — {player}" if player else "")
        return (fig, ts, label)

    # Team bar avg
    if chart == "bar":
        avg = (work.groupby(["display_name"])[metric].mean()
               .reset_index().sort_values(metric, ascending=False).head(n))
        fig = px.bar(avg, x="display_name", y=metric)
        return (fig, avg, f"Average {metric} (top {n})")

    # Heatmap (players x date)
    if chart == "heatmap":
        pivot = (work.pivot_table(index="display_name", columns="date", values=metric, aggfunc="mean"))
        pivot = pivot.sort_index()
        fig = px.imshow(pivot, aspect="auto", origin="lower", color_continuous_scale="Viridis")
        return (fig, pivot.reset_index(), f"Heatmap of {metric}")

    # Default table
    return (None, work[["date","display_name","team","position",metric,"acwr","readiness","flag"]].sort_values(["display_name","date"]),
            f"Table for {metric}")
