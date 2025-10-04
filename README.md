# Dashboard de suivi d’objectifs (Streamlit)

## Lancer
1. Crée un environnement: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. Installe: `pip install -r app/requirements.txt`
3. Démarre: `streamlit run app/streamlit_app.py`

Les données sont stockées dans `data/goals.csv` et `data/progress.csv`.

## Conseils
- Les valeurs de `progress.csv` sont des incréments (ex: +1 livre lu).
- Ajuste les seuils dans `status_from_progress` si besoin.
- Pour héberger, regarde Streamlit Community Cloud, Railway, Render, etc. (attention à la persistance des fichiers).
