import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib import Path
from engine.compute_metrics import compute_team_metrics, daily_summary
from engine.coach_commands import interpret_question

st.set_page_config(page_title="LockerMetrics", layout="wide")
st.title("LockerMetrics")
st.caption("Wearable-agnostic team performance dashboard")

DATA = Path("data")
players_path = DATA/"players.csv"
daily_path   = DATA/"daily_input.csv"

if not players_path.exists() or not daily_path.exists():
    st.warning("Add data/players.csv and data/daily_input.csv to get started.")
    st.stop()

players = pd.read_csv(players_path)
daily   = pd.read_csv(daily_path)

df = daily.merge(players, on="player_id", how="left")
df["date"] = pd.to_datetime(df["date"])
df = compute_team_metrics(df)

latest = df["date"].max()
today  = df[df["date"] == latest] if pd.notna(latest) else df.head(0)

# ===== KPIs
k1,k2,k3,k4 = st.columns(4)
k1.metric("Players", today["player_id"].nunique() if len(today) else 0)
k2.metric("Red Flags", int((today["flag"]=="Red").sum()) if len(today) else 0)
k3.metric("Avg Readiness", f"{np.nanmean(today['readiness']):.0f}" if len(today) else "—")
k4.metric("Avg ACWR", f"{np.nanmean(today['acwr']):.2f}" if len(today) else "—")

# ===== Tabs
tab_readiness, tab_load, tab_recovery, tab_injury, tab_reports, tab_cmd = st.tabs(
    ["Readiness", "Load", "Recovery", "Injury Watch", "Reports", "Coach Command"]
)

# --- Readiness
with tab_readiness:
    st.subheader("Team Readiness (Today)")
    show_cols = ["display_name","team","position","readiness","acwr","sleep_hours","hrv_ms","rhr_bpm","soreness_1_5","flag"]
    if len(today):
        st.dataframe(today.sort_values(["flag","readiness"], ascending=[True, True])[show_cols], use_container_width=True)
    else:
        st.info("No entries for the latest date yet.")

    st.subheader("Readiness Trend")
    if len(df):
        who = st.selectbox("Athlete", df["display_name"].dropna().unique())
        p = df[df["display_name"]==who].sort_values("date")
        fig = px.line(p, x="date", y="readiness", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# --- Load
with tab_load:
    st.subheader("ACWR Trend (Team Average)")
    if len(df):
        team_acwr = (df.groupby("date")["acwr"].mean().reset_index())
        st.line_chart(team_acwr.set_index("date"))

    st.subheader("Workload (RPE x Minutes)")
    if len(df):
        who_l = st.selectbox("Athlete (Load)", df["display_name"].dropna().unique(), key="load_sel")
        p2 = df[df["display_name"]==who_l].sort_values("date")
        fig2 = px.bar(p2, x="date", y="load", labels={"load":"Daily Load"})
        st.plotly_chart(fig2, use_container_width=True)

# --- Recovery
with tab_recovery:
    st.subheader("Sleep / HRV / RHR Trends (Team Avg)")
    if len(df):
        cols = ["sleep_hours","hrv_ms","rhr_bpm"]
        m = df.groupby("date")[cols].mean().reset_index().melt("date", var_name="metric", value_name="value")
        fig3 = px.line(m, x="date", y="value", color="metric", markers=True)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation (last 28 days)")
    if pd.notna(latest):
        cut = df[df["date"] >= (latest - pd.Timedelta(days=28))]
        corr_cols = ["readiness","acwr","sleep_hours","hrv_ms","rhr_bpm","strain_0_21","soreness_1_5","load"]
        corr = cut[corr_cols].corr(numeric_only=True)
        st.dataframe(corr.style.background_gradient(axis=None), use_container_width=True)

# --- Injury Watch
with tab_injury:
    st.subheader("Risk List (Today)")
    if len(today):
        risk = today.sort_values(["flag","readiness","acwr"], ascending=[True, True, False])[["display_name","team","position","readiness","acwr","soreness_1_5","injury_flag","illness_flag","flag"]]
        st.dataframe(risk, use_container_width=True)
    else:
        st.info("No entries for the latest date yet.")

    st.subheader("Return-to-Play Tracker")
    injured = df[df["injury_flag"]==1]["display_name"].dropna().unique().tolist()
    if injured:
        who_i = st.selectbox("Injured athlete", injured)
        p3 = df[df["display_name"]==who_i].sort_values("date")
        fig4 = px.line(p3, x="date", y=["readiness","acwr"], markers=True)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.caption("No injured athletes flagged in data.")

# --- Reports
with tab_reports:
    st.subheader("Coach Report (Preview)")
    st.dataframe(daily_summary(df), use_container_width=True)
    st.write("**Today Snapshot (Top issues)**")
    if len(today):
        risk = today.sort_values(["flag","readiness","acwr"], ascending=[True, True, False])[["display_name","readiness","acwr","flag"]]
        st.dataframe(risk.head(10), use_container_width=True)
    st.caption("PDF export is generated nightly by the GitHub Action (see repo 'reports/' folder).")

# --- Coach Command
with tab_cmd:
    st.subheader("Coach Command Bar")
    help_text = "Examples: 'top 5 readiness today', 'line readiness for JD last 14 days', 'readiness vs acwr last 28 days', 'sql: select display_name, avg(readiness) r from metrics group by 1 order by r desc limit 10'"
    q = st.text_input("Ask a question", placeholder=help_text)
    if q:
        fig, out, msg = interpret_question(q, df)
        st.write(f"**{msg}**")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        if isinstance(out, pd.DataFrame) and len(out):
            st.dataframe(out, use_container_width=True)
