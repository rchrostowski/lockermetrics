import pandas as pd, numpy as np

METRIC_COLUMNS = [
    "sleep_hours","sleep_quality_1_5","hrv_ms","rhr_bpm","strain_0_21",
    "session_rpe_0_10","session_minutes","soreness_1_5","body_weight_kg"
]

def _rolling_player(df, col, win, minp):
    return (df
            .groupby("player_id", group_keys=False)[col]
            .rolling(win, min_periods=minp).mean()
            .reset_index(level=0, drop=True))

def compute_team_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["player_id","date"])

    # ---- Basic load
    df["load"] = df["session_rpe_0_10"].fillna(0) * df["session_minutes"].fillna(0)

    # ---- Rolling loads
    df["roll7"]  = _rolling_player(df, "load", 7, 3)
    df["roll28"] = _rolling_player(df, "load", 28, 7)
    df["acwr"]   = (df["roll7"] / df["roll28"]).replace([np.inf, -np.inf], np.nan)

    # ---- Baseline z-scores (28d) per player
    def z_player(col, higher_is_better=True):
        g = df.groupby("player_id")[col]
        mu  = g.transform(lambda s: s.rolling(28, min_periods=7).mean())
        std = g.transform(lambda s: s.rolling(28, min_periods=7).std())
        z = (df[col] - mu) / std
        if not higher_is_better: z = -z
        return z

    df["z_hrv"]    = z_player("hrv_ms", True)
    df["z_sleep"]  = z_player("sleep_hours", True)
    df["z_rhr"]    = z_player("rhr_bpm", False)
    df["z_strain"] = z_player("strain_0_21", False)
    df["z_sore"]   = z_player("soreness_1_5", False)

    # ---- Readiness (0–100 scaled daily by team)
    w = dict(z_hrv=0.30, z_sleep=0.20, z_rhr=0.20, z_strain=0.15, z_sore=0.15)
    r_raw = (df["z_hrv"].fillna(0)*w["z_hrv"] + df["z_sleep"].fillna(0)*w["z_sleep"]
            + df["z_rhr"].fillna(0)*w["z_rhr"] + df["z_strain"].fillna(0)*w["z_strain"]
            + df["z_sore"].fillna(0)*w["z_sore"])
    df["readiness_raw"] = r_raw

    def scale_today(group):
        vals = group["readiness_raw"].to_numpy()
        if len(vals) > 5:
            lo, hi = np.nanpercentile(vals, [5,95])
        else:
            lo, hi = np.nanmin(vals), np.nanmax(vals)
        rng = (hi - lo) if (hi - lo) != 0 else 1.0
        group["readiness"] = ((vals - lo) / rng * 100).clip(0, 100)
        return group

    df = df.groupby("date", group_keys=False).apply(scale_today)

    # ---- Flags
    df["flag"] = "Green"
    df.loc[(df["acwr"] > 1.3) | (df["readiness"] < 50), "flag"] = "Yellow"
    df.loc[(df["acwr"] > 1.5) | (df["readiness"] < 30) | (df["injury_flag"] == 1), "flag"] = "Red"

    return df

def daily_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    latest = df_all["date"].max()
    today  = df_all[df_all["date"] == latest]
    out = {
        "players": int(today["player_id"].nunique()) if len(today) else 0,
        "reds": int((today["flag"] == "Red").sum()) if len(today) else 0,
        "avg_ready": float(np.nanmean(today["readiness"])) if len(today) else np.nan,
        "avg_acwr": float(np.nanmean(today["acwr"])) if len(today) else np.nan,
        "date": str(latest.date()) if pd.notna(latest) else ""
    }
    return pd.DataFrame([out])
