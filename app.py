import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

st.title("🏥 Le Diagnostiqueur Médical")
st.write("Renseignez les mesures cellulaires pour obtenir une estimation.")

# ─── Charger le modèle une seule fois ────────────────────────────
@st.cache_resource
def charger_modele() :
    return joblib.load("modele_diagnostic_cancer.pkl")

modele = charger_modele()

# ─── Récupérer les noms et moyennes de toutes les features ───────
data = load_breast_cancer()
toutes_features = pd.DataFrame(data.data, columns=data.feature_names)
moyennes = toutes_features.mean()

# ─── Formulaire pour les 5 features les plus importantes ─────────
st.header("Mesures principales")

worst_concave_points = st.number_input(
    "Worst concave points", min_value=0.0, max_value=0.3, value=float(moyennes["worst concave points"])
)
worst_perimeter = st.number_input("Worst perimeter", 
     50.0, 260.0, float(moyennes["worst perimeter"])       # On peut aussi faire sans min_value=..., max_value=..., value=...
)
mean_concave_points = st.number_input(
    "Mean concave points", 0.0, 0.2, float(moyennes["mean concave points"])
)
worst_radius = st.number_input(
    "Worst radius", 7.0, 40.0, float(moyennes["worst radius"])
)
mean_perimeter = st.number_input(
    "Mean perimeter", 40.0, 190.0, float(moyennes["mean perimeter"])
)


# ─── Construire l'observation complète ───────────────────────────

entrée = moyennes.copy()

entrée["worst concave points"] = worst_concave_points
entrée["worst perimeter"]      = worst_perimeter
entrée["mean concave points"]  = mean_concave_points
entrée["worst radius"]         = worst_radius
entrée["mean perimeter"]       = mean_perimeter

X_utilisateur = pd.DataFrame([entrée])


# ─── Effectuer la prédiction ──────────────────────────────────────

if st.button("Obtenir le diagnostic"):
    prédiction = modele.predict(X_utilisateur)[0]
    probabilités = modele.predict_proba(X_utilisateur)[0]

    if prédiction == 0:
        st.error(f"⚠️ Diagnostic prédit : MALIN (probabilité : {probabilités[0]:.1%})")
    else:
        st.success(f"✅ Diagnostic prédit : BÉNIN (probabilité : {probabilités[1]:.1%})")

    st.write("Détail des probabilités :")
    st.write(f"- Malin : {probabilités[0]:.1%}")
    st.write(f"- Bénin : {probabilités[1]:.1%}")

    st.caption("⚠️ Cet outil est une aide à la décision et ne remplace en aucun cas un diagnostic médical professionnel.")
    if probabilités[0] < 0.7 and probabilités[1] < 0.7 :
        st.warning("⚠️ Confiance faible : une évaluation médicale complémentaire est recommandée.")

# Ajoutez une condition : si la probabilité de la classe prédite est inférieure à 70%, 
# affichez un st.warning recommandant un examen complémentaire.

