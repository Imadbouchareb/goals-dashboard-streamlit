import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

st.set_page_config(page_title="Suivi d'Objectifs", layout="wide", page_icon="✅")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOALS_PATH = DATA_DIR / "goals.csv"
PROG_PATH = DATA_DIR / "progress.csv"

@st.cache_data
def load_goals():
    df = pd.read_csv(GOALS_PATH, dtype=str)
    # Types
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
    for col in ["target_value", "weight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

@st.cache_data
def load_progress():
    df = pd.read_csv(PROG_PATH, dtype=str)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df

def ensure_data_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not GOALS_PATH.exists():
        sample_goals = pd.DataFrame([
            {
                "goal_id": "G1",
                "goal_name": "Perdre 5 kg",
                "start_date": "2025-10-01",
                "end_date": "2026-01-01",
                "target_value": "5",
                "unit": "kg",
                "owner": "Imadbouchareb",
                "category": "Santé",
                "weight": "1",
                "color": "#e74c3c",
                "notes": "Objectif santé Q4"
            },
            {
                "goal_id": "G2",
                "goal_name": "Lire 12 livres",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "target_value": "12",
                "unit": "livres",
                "owner": "Imadbouchareb",
                "category": "Apprentissage",
                "weight": "1",
                "color": "#3498db",
                "notes": "Routine lecture"
            },
        ])
        sample_goals.to_csv(GOALS_PATH, index=False)
    if not PROG_PATH.exists():
        sample_progress = pd.DataFrame([
            # Incréments de lecture (total = 6)
            {"date": "2025-02-01", "goal_id": "G2", "value": "1", "note": "Premier livre"},
            {"date": "2025-03-15", "goal_id": "G2", "value": "1", "note": "Deuxième livre"},
            {"date": "2025-05-30", "goal_id": "G2", "value": "2", "note": "Deux livres en mai"},
            {"date": "2025-08-20", "goal_id": "G2", "value": "2", "note": "Deux livres en août"},
        ])
        sample_progress.to_csv(PROG_PATH, index=False)

def status_from_progress(actual_pct: float, expected_pct: float) -> str:
    # Seuils: >=95% attendu => On Track; >=70% => At Risk; sinon Off Track
    if actual_pct >= expected_pct * 0.95:
        return "On Track"
    if actual_pct >= expected_pct * 0.70:
        return "At Risk"
    return "Off Track"

def clamp(v, lo, hi=None):
    if hi is None:
        return max(lo, v)
    return max(lo, min(hi, v))

def compute_summary(goals_df: pd.DataFrame, progress_df: pd.DataFrame, today: date):
    # Somme des incréments réalisés par objectif
    done = progress_df.groupby("goal_id")["value"].sum().rename("actual_value")
    df = goals_df.merge(done, how="left", left_on="goal_id", right_index=True)
    df["actual_value"] = df["actual_value"].fillna(0.0)

    # Calculs temporels
    total_days = (pd.to_datetime(df["end_date"]) - pd.to_datetime(df["start_date"])).dt.days.clip(lower=0)
    elapsed_days = (pd.to_datetime(pd.Series([today]*len(df))) - pd.to_datetime(df["start_date"])).dt.days
    elapsed_days = elapsed_days.apply(lambda x: clamp(x, 0, None))
    elapsed_days = np.where((pd.to_datetime([today]*len(df)) > pd.to_datetime(df["end_date"])), total_days, elapsed_days)

    df["total_days"] = total_days.replace(0, 1)  # évite /0 si même jour
    df["elapsed_days"] = elapsed_days

    # Attendu (linéaire)
    df["expected_pct"] = (df["elapsed_days"] / df["total_days"]).astype(float).clip(0, 1)
    # Réel
    df["progress_pct"] = (df["actual_value"] / df["target_value"]).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1)

    df["status"] = [status_from_progress(a, e) for a, e in zip(df["progress_pct"], df["expected_pct"])]

    # Jours restants
    df["days_left"] = (pd.to_datetime(df["end_date"]) - pd.to_datetime([today]*len(df))).dt.days.clip(lower=0)

    # Vélocités
    df["velocity"] = df["actual_value"] / df["elapsed_days"].replace(0, np.nan)
    df["velocity"] = df["velocity"].fillna(0.0)
    remaining = (df["target_value"] - df["actual_value"]).clip(lower=0)
    df["needed_velocity"] = remaining / df["days_left"].replace(0, np.nan)
    df["needed_velocity"] = df["needed_velocity"].fillna(np.inf)

    # Score global pondéré
    total_weight = df["weight"].sum()
    if total_weight <= 0:
        total_weight = 1.0
    df["weighted_progress"] = df["progress_pct"] * df["weight"] / total_weight

    return df

def cumulative_series_for_goal(progress_df: pd.DataFrame, goal_row: pd.Series, today: date):
    # Série cumulée réelle par date jusqu'à today
    g = progress_df[progress_df["goal_id"] == goal_row["goal_id"]].copy()
    if g.empty:
        return pd.DataFrame({
            "date": [goal_row["start_date"]],
            "actual_cum": [0.0],
            "expected_cum": [0.0],
        })
    g = g.groupby("date")["value"].sum().sort_index().cumsum().rename("actual_cum").to_frame()
    # Génère une série de dates complètes
    end_for_chart = min(goal_row["end_date"], today)
    date_index = pd.date_range(goal_row["start_date"], end_for_chart, freq="D").date
    df = pd.DataFrame({"date": date_index}).merge(g, left_on="date", right_index=True, how="left")
    df["actual_cum"] = df["actual_cum"].ffill().fillna(0.0)
    # Attendu linéaire
    total_days = max((goal_row["end_date"] - goal_row["start_date"]).days, 1)
    elapsed = [(d - goal_row["start_date"]).days for d in df["date"]]
    expected_pct = np.array(elapsed) / total_days
    expected_pct = np.clip(expected_pct, 0, 1)
    df["expected_cum"] = expected_pct * goal_row["target_value"]
    return df

def save_goals(df: pd.DataFrame):
    # Invalide le cache après écriture
    df.to_csv(GOALS_PATH, index=False)
    load_goals.clear()

def append_progress(row: dict):
    df = load_progress()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(PROG_PATH, index=False)
    load_progress.clear()

# Initialisation
ensure_data_files()
goals_df = load_goals()
progress_df = load_progress()
today = date.today()

# Sidebar: Filtres et Ajouts
with st.sidebar:
    st.header("Filtres")
    owners = sorted(goals_df["owner"].dropna().unique().tolist())
    cats = sorted(goals_df["category"].dropna().unique().tolist())
    sel_owners = st.multiselect("Propriétaire(s)", owners, default=owners)
    sel_cats = st.multiselect("Catégorie(s)", cats, default=cats)

    st.divider()
    st.header("Ajouter une entrée")
    with st.form("add_progress_form", clear_on_submit=True):
        goal_choices = goals_df[(goals_df["owner"].isin(sel_owners)) & (goals_df["category"].isin(sel_cats))]
        goal_map = {f'{r["goal_name"]} ({r["goal_id"]})': r["goal_id"] for _, r in goal_choices.iterrows()}
        sel_goal_label = st.selectbox("Objectif", options=list(goal_map.keys()) if goal_map else [])
        the_date = st.date_input("Date", value=today)
        the_value = st.number_input("Valeur (incrément)", min_value=0.0, step=0.1)
        the_note = st.text_input("Note (facultatif)")
        submitted = st.form_submit_button("Ajouter")
        if submitted and sel_goal_label:
            append_progress({"date": the_date, "goal_id": goal_map[sel_goal_label], "value": the_value, "note": the_note})
            st.success("Entrée ajoutée. Rafraîchis la page si besoin.")
    st.divider()
    st.header("Créer un objectif")
    with st.form("add_goal_form", clear_on_submit=True):
        new_id = st.text_input("ID (unique, ex: G3)")
        new_name = st.text_input("Nom")
        col1, col2 = st.columns(2)
        with col1:
            new_start = st.date_input("Début", value=today)
        with col2:
            new_end = st.date_input("Fin", value=today)
        target = st.number_input("Cible (nombre)", min_value=0.0, step=0.1)
        unit = st.text_input("Unité (ex: kg, livres, km)")
        owner = st.text_input("Propriétaire", value="Imadbouchareb")
        cat = st.text_input("Catégorie", value="Général")
        weight = st.number_input("Poids (pondération)", min_value=0.0, value=1.0, step=0.1)
        color = st.color_picker("Couleur", value="#2ecc71")
        notes = st.text_input("Notes")
        add_goal = st.form_submit_button("Créer")
        if add_goal:
            if not new_id or goals_df["goal_id"].eq(new_id).any():
                st.error("ID manquant ou déjà utilisé.")
            else:
                new_row = pd.DataFrame([{
                    "goal_id": new_id,
                    "goal_name": new_name,
                    "start_date": new_start,
                    "end_date": new_end,
                    "target_value": target,
                    "unit": unit,
                    "owner": owner,
                    "category": cat,
                    "weight": weight,
                    "color": color,
                    "notes": notes
                }])
                new_df = pd.concat([goals_df, new_row], ignore_index=True)
                save_goals(new_df)
                st.success("Objectif créé. Rafraîchis la page si besoin.")

# Filtrage
mask = goals_df["owner"].isin(sel_owners) & goals_df["category"].isin(sel_cats)
goals_view = goals_df[mask].copy()

summary = compute_summary(goals_view, progress_df, today)

st.title("Dashboard de suivi d'avancement")
st.caption("Suivi linéaire vs. attendu. Ajoute tes entrées dans la barre latérale.")

# KPIs
colA, colB, colC, colD = st.columns(4)
overall = summary["weighted_progress"].sum() if not summary.empty else 0.0
on_track = (summary["status"] == "On Track").sum()
at_risk = (summary["status"] == "At Risk").sum()
off_track = (summary["status"] == "Off Track").sum()
with colA:
    st.metric("Progression globale", f"{overall*100:.0f}%")
with colB:
    st.metric("On Track", f"{on_track}")
with colC:
    st.metric("At Risk", f"{at_risk}")
with colD:
    st.metric("Off Track", f"{off_track}")

st.divider()

# Détails par objectif
if summary.empty:
    st.info("Aucun objectif à afficher avec ces filtres.")
else:
    for _, row in summary.sort_values(by="end_date").iterrows():
        with st.expander(f'{row["goal_name"]} ({row["goal_id"]}) — {row["status"]}', expanded=True):
            c1, c2, c3 = st.columns([2,2,1])
            with c1:
                st.write(f'Période: {row["start_date"]} → {row["end_date"]}  |  Reste: {int(row["days_left"])} j')
                st.progress(float(row["progress_pct"]))
                st.caption(f'Réel: {row["actual_value"]:.2f}/{row["target_value"]:.2f} {row["unit"]} | Attendu: {row["expected_pct"]*100:.0f}%')
            with c2:
                st.write("Vélocité")
                st.metric("Réelle/jour", f'{row["velocity"]:.2f} {row["unit"]}/j')
                needed = row["needed_velocity"]
                needed_txt = "∞" if np.isinf(needed) else f'{needed:.2f} {row["unit"]}/j'
                st.metric("Nécessaire/jour", needed_txt)
            with c3:
                st.write("")
                if row["status"] == "On Track":
                    st.success("Bien joué, continue ! 💪")
                elif row["status"] == "At Risk":
                    st.warning("Ralentissement: petit coup d'accélérateur 🚀")
                else:
                    st.error("Hors trajectoire: ajuste le plan aujourd'hui 🔧")

            # Graphique
            series = cumulative_series_for_goal(progress_df, row, today)
            chart_df = series.melt(id_vars="date", value_vars=["actual_cum", "expected_cum"], var_name="type", value_name="val")
            import altair as alt
            line = alt.Chart(chart_df).mark_line().encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("val:Q", title=f'Cumul ({row["unit"]})'),
                color=alt.Color("type:N", title="Série", scale=alt.Scale(domain=["actual_cum","expected_cum"], range=[row["color"], "#888888"]))
            ).properties(height=250)
            st.altair_chart(line, use_container_width=True)

st.caption("Note: les données sont stockées en CSV localement (dossier data/). Sur un hébergeur gratuit, la persistance disque peut être limitée.")