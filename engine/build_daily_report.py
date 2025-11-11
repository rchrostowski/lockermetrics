from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

def build_pdf(df, path="reports/daily_report.pdf"):
    latest = df["date"].max()
    today = df[df["date"] == latest]

    c = canvas.Canvas(path, pagesize=LETTER)
    w, h = LETTER

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, h-50, f"LockerMetrics — Daily Report ({latest.date()})")

    c.setFont("Helvetica", 11)
    txt = (
        f"Players: {today['player_id'].nunique()} | "
        f"Reds: {(today['flag']=='Red').sum()} | "
        f"Avg Readiness: {today['readiness'].mean():.0f} | "
        f"Avg ACWR: {today['acwr'].mean():.2f}"
    )
    c.drawString(50, h-75, txt)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, h-110, "Top Risks:")
    y = h-130
    risk = today.sort_values(["flag","readiness","acwr"], ascending=[True, True, False])[["display_name","readiness","acwr","flag"]].head(10)
    for _, r in risk.iterrows():
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"{r['display_name']}: Readiness {r['readiness']:.0f}, ACWR {r['acwr']:.2f}, {r['flag']}")
        y -= 16
        if y < 80:
            c.showPage(); y = h - 80

    c.showPage()
    c.save()
    return path
