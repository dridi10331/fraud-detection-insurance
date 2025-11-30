import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Détection de Fraude Assurance", page_icon="🚗", layout='centered')

st.title("🚗 Détection de Fraude d'Assurance - Esprit IA")

st.markdown("""
Bienvenue sur l'interface de démonstration du projet de détection de fraude assurance.  
Choisissez une section :
- 📊 **Visualiser les statistiques du dataset**
- 🚨 **Voir la liste des fraudes détectées**
- 🧪 **Tester la prédiction d'un dossier**
""")

# Charger les données nécessaires
@st.cache_data
def load_fraudes():
    return pd.read_csv('fraudes_detectees_enrichi.csv', encoding='utf-8-sig')

@st.cache_data
def load_resume():
    return pd.read_csv('resume_types_fraudes.csv', encoding='utf-8-sig')

frauds = load_fraudes()
resume = load_resume()

tab1, tab2, tab3 = st.tabs(["📊 Statistiques globales", "🚨 Fraudes détectées", "🧪 Tester une prédiction"])

with tab1:
    st.header("📈 Types de fraudes détectées")
    st.write(f"Nombre total de fraudes détectées : **{len(frauds)}**")
    fig, ax = plt.subplots()
    ax.pie(resume['NOMBRE_CAS'], labels=resume['TYPE_FRAUDE'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("Répartition par type")
    st.dataframe(resume[['TYPE_FRAUDE', 'NOMBRE_CAS', 'POURCENTAGE', 'DESCRIPTION']])

with tab2:
    st.header("🚨 Liste des fraudes détectées")
    st.write(f"Affichage des {len(frauds)} cas détectés")
    st.dataframe(frauds[['ID_SINISTRE','FRAUD_TYPE','NOMBRE_PATTERNS','NIVEAU_RISQUE','DETAILS']].head(50))
    export = st.download_button("📥 Télécharger tout le fichier CSV", data=frauds.to_csv(index=False).encode('utf-8-sig'),
                               file_name='fraudes_detectees_enrichi.csv', mime='text/csv')

with tab3:
    st.header("🧪 Tester un dossier de sinistre")
    st.write("Remplis les informations pour obtenir une prédiction (exemple simplifié)")

    days_to_declare = st.slider("Délai (jours entre sinistre et déclaration)", 0, 90, 10)
    late_declaration = st.selectbox("Déclaration tardive (> 30 jours)?", ["Non","Oui"])
    vague_location = st.selectbox("Localisation vague ('XX', '*',...)?", ["Non","Oui"])
    expert_freq = st.slider("Expert: nombre de cas traités", 0, 100, 10)
    
    pattern_count = 0
    risk_factors = []
    if days_to_declare > 30:
        pattern_count += 1
        risk_factors.append("Déclaration tardive (>30j)")
    if late_declaration == "Oui":
        pattern_count += 1
        risk_factors.append("Déclaration déclarée tardivement")
    if vague_location == "Oui":
        pattern_count += 1
        risk_factors.append("Localisation vague/suspecte")
    if expert_freq > 50:
        pattern_count += 1
        risk_factors.append("Expert très fréquent (>50 cas)")

    if st.button("Prédire"):
        if pattern_count == 0:
            st.success("🟢 Dossier LÉGITIME, faible risque de fraude.")
        elif pattern_count == 1:
            st.warning("🟡 Risque MOYEN de fraude: patterns détectés: " + ", ".join(risk_factors))
        else:
            st.error("🔴 Risque ÉLEVÉ de fraude! Patterns: " + ", ".join(risk_factors))
        st.markdown(f"**Facteurs détectés:** {', '.join(risk_factors) if risk_factors else 'Aucun'}")

st.markdown("---")
st.write("Projet Esprit AI · 2025 — Contact : étudiant.esprit@esprit.tn")
