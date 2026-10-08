import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Budget Variance Analyzer", page_icon="📊", layout="wide")

# Check Google AI Studio for the current free model name if this one stops working.
GEMINI_MODEL = "gemini-3-flash-preview"
REQUIRED_COLS = ["Department", "Line_Item", "Budget", "Actual"]


# ---------- Data ----------
@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv("sample_budget_data.csv")


def analyze(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["Variance"] = df["Actual"] - df["Budget"]
    df["Variance_%"] = (df["Variance"] / df["Budget"] * 100).round(1)
    df["Status"] = "On track"
    df.loc[df["Variance_%"] > threshold, "Status"] = "Over budget"
    df.loc[df["Variance_%"] < -threshold, "Status"] = "Under budget"
    return df


def top_outliers(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return df.reindex(df["Variance"].abs().sort_values(ascending=False).index).head(n)


# ---------- AI commentary ----------
def rule_based_commentary(df: pd.DataFrame, outliers: pd.DataFrame) -> str:
    tb, ta = df["Budget"].sum(), df["Actual"].sum()
    diff = ta - tb
    direction = "over" if diff > 0 else "under"
    lines = [
        f"Overall spend is {abs(diff):,.0f} {direction} budget "
        f"({diff / tb * 100:+.1f}%) — actual {ta:,.0f} vs budget {tb:,.0f}.",
        "Largest deviations:",
    ]
    for _, r in outliers.iterrows():
        lines.append(
            f"- {r['Department']} / {r['Line_Item']}: {r['Variance']:+,.0f} ({r['Variance_%']:+.1f}%)"
        )
    return "\n".join(lines)


def ai_commentary(df: pd.DataFrame, outliers: pd.DataFrame, api_key: str) -> str:
    dept = (
        df.groupby("Department")[["Budget", "Actual", "Variance"]].sum().round(0).to_string()
    )
    prompt = f"""You are a finance analyst writing for a non-finance manager.
Using ONLY the numbers below, write a concise variance commentary (max 150 words):
1) overall position, 2) the 3 biggest outliers with likely business reasons phrased as
hypotheses to verify (not facts), 3) two recommended actions.

Totals: Budget={df['Budget'].sum():,.0f}, Actual={df['Actual'].sum():,.0f}
By department:
{dept}

Top 3 outlier line items:
{outliers[['Department','Line_Item','Budget','Actual','Variance','Variance_%']].to_string(index=False)}
"""
    last_error = None
    for model in [GEMINI_MODEL, "gemini-flash-latest", "gemini-flash-lite-latest"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for _ in range(2):
            try:
                resp = requests.post(
                    url,
                    params={"key": api_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                last_error = e
                time.sleep(2)
    raise last_error


def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return st.sidebar.text_input("Gemini API key (optional)", type="password")


def clean(df: pd.DataFrame):
    """Convert Budget/Actual to numbers and drop unusable rows."""
    df = df.copy()
    for c in ["Budget", "Actual"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[REQUIRED_COLS].isna().any(axis=1) | (df["Budget"] <= 0)
    return df[~bad].reset_index(drop=True), int(bad.sum())


# ---------- UI ----------
st.title("📊 Budget Variance Analyzer")
st.caption("Upload actuals vs budget → get variances, top outliers and AI-written commentary.")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    st.caption("Required columns: " + ", ".join(REQUIRED_COLS))
    threshold = st.slider("Flag variance beyond ±%", 1, 30, 10)
    api_key = get_api_key()

if uploaded:
    raw = pd.read_csv(uploaded)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
else:
    raw = load_sample()
    st.info("Showing built-in sample data. Upload your own CSV from the sidebar.")

raw, n_bad = clean(raw)
if n_bad:
    st.warning(f"{n_bad} row(s) skipped: blank, non-numeric, or zero/negative budget values.")
if raw.empty:
    st.error("No valid rows found. Check that Budget and Actual contain numbers.")
    st.stop()

# Reset saved commentary whenever the data changes
sig = int(pd.util.hash_pandas_object(raw, index=False).sum())
if st.session_state.get("sig") != sig:
    st.session_state["sig"] = sig
    st.session_state.pop("commentary", None)

data = analyze(raw, threshold)
outliers = top_outliers(data)

# KPIs
tb, ta = data["Budget"].sum(), data["Actual"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Budget", f"{tb:,.0f}")
c2.metric("Total Actual", f"{ta:,.0f}")
c3.metric("Net Variance", f"{ta - tb:+,.0f}", f"{(ta - tb) / tb * 100:+.1f}%", delta_color="inverse")
c4.metric("Items flagged", int((data["Status"] != "On track").sum()))

# Charts
left, right = st.columns(2)
dept = data.groupby("Department")[["Budget", "Actual"]].sum().reset_index()
left.plotly_chart(
    px.bar(dept, x="Department", y=["Budget", "Actual"], barmode="group", title="Budget vs Actual by Department"),
    width="stretch",
)
right.plotly_chart(
    px.bar(
        data.sort_values("Variance_%"),
        x="Variance_%",
        y="Line_Item",
        color="Status",
        orientation="h",
        title="Variance % by Line Item",
        color_discrete_map={"Over budget": "#d62728", "Under budget": "#1f77b4", "On track": "#2ca02c"},
    ),
    width="stretch",
)

# Outliers
st.subheader("🚩 Top 3 outlier line items")
st.dataframe(
    outliers[["Department", "Line_Item", "Budget", "Actual", "Variance", "Variance_%", "Status"]],
    width="stretch",
    hide_index=True,
)

# Commentary
st.subheader("🤖 Variance commentary")
st.caption(
    "Privacy: clicking Generate sends a summary (totals, department totals and the top 3 "
    "outlier rows) to Google's Gemini API. Do not upload confidential data."
)
if st.button("Generate commentary", type="primary"):
    with st.spinner("Analyzing..."):
        if api_key:
            try:
                text = ai_commentary(data, outliers, api_key).replace("$", "\\$")
                st.session_state["commentary"] = ("ai", text, "")
            except Exception as e:
                detail = ""
                if getattr(e, "response", None) is not None:
                    detail = f"{e.response.status_code}"
                msg = f"AI call failed ({type(e).__name__} {detail}). Showing rule-based commentary instead."
                st.session_state["commentary"] = ("fallback", msg, rule_based_commentary(data, outliers))
        else:
            msg = "No API key provided — using rule-based commentary."
            st.session_state["commentary"] = ("nokey", msg, rule_based_commentary(data, outliers))

saved = st.session_state.get("commentary")
if saved:
    kind, text, extra = saved
    if kind == "ai":
        st.markdown(text)
    else:
        if kind == "fallback":
            st.warning(text)
        else:
            st.caption(text)
        st.text(extra)
    st.caption("AI text is a draft. Verify reasons with department owners before acting.")

with st.expander("Full data table"):
    st.dataframe(data, width="stretch", hide_index=True)
    st.download_button("Download analysis CSV", data.to_csv(index=False), "variance_analysis.csv")
