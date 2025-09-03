import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import sqlite3
import requests
import json
from io import BytesIO
# Configuration de la page
st.set_page_config(
    page_title="Currency Risk Manager - Excis Compliance",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Initialisation de la base de données SQLite
def init_db():
    conn = sqlite3.connect('currency_risk.db')
    c = conn.cursor()
   
    # Création de la table des expositions si elle n'existe pas
    c.execute('''CREATE TABLE IF NOT EXISTS exposure
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  currency TEXT NOT NULL,
                  exposure_before REAL NOT NULL,
                  exposure_after REAL NOT NULL,
                  volatility REAL NOT NULL,
                  last_updated TEXT NOT NULL)''')
   
    # Création de la table des taux de change si elle n'existe pas
    c.execute('''CREATE TABLE IF NOT EXISTS exchange_rates
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  currency TEXT NOT NULL,
                  rate_to_gbp REAL NOT NULL,
                  last_updated TEXT NOT NULL)''')
   
    conn.commit()
    conn.close()
# Fonction pour récupérer les taux de change en temps réel
def get_exchange_rates():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/GBP", timeout=10)
        rates = response.json()["rates"]
        return rates
    except Exception as e:
        st.error(f"Erreur lors de la récupération des taux de change: {e}")
        return None
# Fonction pour mettre à jour les taux de change dans la base de données
def update_exchange_rates_db():
    rates = get_exchange_rates()
    if rates:
        conn = sqlite3.connect('currency_risk.db')
        c = conn.cursor()
       
        # Vider la table avant d'insérer de nouvelles données
        c.execute("DELETE FROM exchange_rates")
       
        # Insérer les nouveaux taux
        for currency, rate in rates.items():
            c.execute('''INSERT INTO exchange_rates (currency, rate_to_gbp, last_updated)
                         VALUES (?, ?, ?)''',
                      (currency, rate, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
       
        conn.commit()
        conn.close()
        return True
    return False
# Fonction pour récupérer les taux de change depuis la base de données
def get_exchange_rates_db():
    conn = sqlite3.connect('currency_risk.db')
    c = conn.cursor()
    c.execute("SELECT currency, rate_to_gbp FROM exchange_rates")
    rates = dict(c.fetchall())
    conn.close()
    return rates
# Fonction pour récupérer les données d'exposition depuis la base de données
def get_exposure_data():
    conn = sqlite3.connect('currency_risk.db')
    c = conn.cursor()
    c.execute("SELECT currency, exposure_before, exposure_after, volatility FROM exposure")
    data = c.fetchall()
    conn.close()
   
    if data:
        return pd.DataFrame(data, columns=['Currency', 'Exposure Before (GBP)', 'Exposure After (GBP)', 'Volatility (%)'])
    return None
# Fonction pour mettre à jour les données d'exposition dans la base de données
def update_exposure_data_db(currency_data):
    conn = sqlite3.connect('currency_risk.db')
    c = conn.cursor()
   
    # Vider la table avant d'insérer de nouvelles données
    c.execute("DELETE FROM exposure")
   
    # Insérer les nouvelles données
    for currency, before, after, volatility in currency_data:
        c.execute('''INSERT INTO exposure (currency, exposure_before, exposure_after, volatility, last_updated)
                     VALUES (?, ?, ?, ?, ?)''',
                  (currency, before, after, volatility, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
   
    conn.commit()
    conn.close()
# Fonction pour sauvegarder le DataFrame actuel dans SQLite
def save_current_data_to_sqlite(df):
    data = [tuple(row) for row in df[['Currency', 'Exposure Before (GBP)', 'Exposure After (GBP)', 'Volatility (%)']].values]
    update_exposure_data_db(data)
# Initialiser la base de données
init_db()
# Données initiales basées sur le document
initial_data = {
    "Total Exposure (GBP)": [2_000_000, 300_000],
    "Potential Loss (GBP)": [200_000, 30_000],
    "USD Exposure (%)": [65, 65],
    "EUR Exposure (%)": [30, 30],
    "Other Exposure (%)": [5, 5],
    "Implementation Cost (GBP)": [0, 63_500],
    "Net Gain (GBP)": [0, 136_500],
    "ROI (%)": [0, 229],
    "Payback (months)": [0, 3.6]
}
# Création du DataFrame
df = pd.DataFrame(initial_data, index=["Before Intervention", "After Intervention"])
# Données des devises
currency_data = [
    ("USD", 1_300_000, 195_000, 12),
    ("EUR", 600_000, 90_000, 8),
    ("AED", 13_664, 5_000, 5),
    ("CAD", 50_000, 0, 4),
    ("SGD", 36_336, 0, 3)
]
# Initialiser session_state pour les données d'exposition
if 'exposure_df' not in st.session_state:
    exposure_df = get_exposure_data()
    if exposure_df is None:
        update_exposure_data_db(currency_data)
        exposure_df = pd.DataFrame(currency_data, columns=['Currency', 'Exposure Before (GBP)', 'Exposure After (GBP)', 'Volatility (%)'])
    st.session_state.exposure_df = exposure_df
# Données des solutions
solutions_data = {
    "Solution": [
        "50% Hedging",
        "Interco Restructuring",
        "Real-Time Monitoring",
        "Currency Diversification",
        "Training/Support"
    ],
    "Cost (GBP)": [18_000, 25_000, 30_000, 5_000, 5_000],
    "Reduced Exposure (GBP)": [1_000_000, 800_000, 200_000, 150_000, 50_000],
    "Avoided Loss (GBP)": [150_000, 40_000, 50_000, 15_000, 5_000],
    "ROI (%)": [733, 160, 167, 200, 100],
    "Timeline (days)": [30, 45, 30, 90, 60]
}
solutions_df = pd.DataFrame(solutions_data)
# Fonction pour calculer les scénarios
def calculate_scenario(usd_eur_change):
    # Calcul basé sur les données du document
    base_exposure = 300_000 # Après intervention
    usd_share = 0.65
    eur_share = 0.30
   
    # Calcul de l'impact
    usd_impact = base_exposure * usd_share * (usd_eur_change / 100)
    eur_impact = base_exposure * eur_share * (usd_eur_change / 100)
    total_impact = usd_impact + eur_impact
   
    # Application des protections (hedging)
    hedged_protection = abs(total_impact) * 0.75 # 75% de protection comme dans le document
   
    return {
        "Total Impact (GBP)": total_impact,
        "After Protection (GBP)": total_impact - hedged_protection,
        "Protection Applied (GBP)": hedged_protection
    }
# Interface utilisateur
st.title("💱 Currency Risk Manager - Excis Compliance Ltd")
st.markdown("*Comprehensive Solution for Managing Currency Fluctuation Risk*")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%B %d, %Y')}")
# Sidebar pour les contrôles
st.sidebar.header("Control Panel")
# Bouton pour mettre à jour les taux de change
if st.sidebar.button("🔄 Update Exchange Rates"):
    if update_exchange_rates_db():
        st.sidebar.success("Exchange rates updated successfully!")
    else:
        st.sidebar.error("Failed to update exchange rates.")
# Sélection de la source de données
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["SQLite", "CSV", "Excel"],
    index=0
)
# Gestion des fichiers uploadés
uploaded_file = None
if data_source == "CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
elif data_source == "Excel":
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
# Traitement des fichiers uploadés
if uploaded_file is not None:
    try:
        if data_source == "CSV":
            new_data = pd.read_csv(uploaded_file)
        else: # Excel
            new_data = pd.read_excel(uploaded_file)
       
        # Vérifier les colonnes requises
        required_columns = ['Currency', 'Exposure Before (GBP)', 'Exposure After (GBP)', 'Volatility (%)']
        if all(col in new_data.columns for col in required_columns):
            st.session_state.exposure_df = new_data[required_columns]
            st.sidebar.success(f"Data loaded from {data_source} successfully!")
        else:
            st.sidebar.error(f"File must contain columns: {', '.join(required_columns)}")
    except Exception as e:
        st.sidebar.error(f"Error loading {data_source} file: {str(e)}")
# Bouton pour sauvegarder les données actuelles dans SQLite
if st.sidebar.button("💾 Save Current Data to SQLite"):
    save_current_data_to_sqlite(st.session_state.exposure_df)
    st.sidebar.success("Data saved to SQLite successfully!")
# Bouton pour réinitialiser les données d'exposition
if st.sidebar.button("🔄 Reset Exposure Data"):
    update_exposure_data_db(currency_data)
    st.session_state.exposure_df = pd.DataFrame(currency_data, columns=['Currency', 'Exposure Before (GBP)', 'Exposure After (GBP)', 'Volatility (%)'])
    st.sidebar.success("Exposure data reset to initial values!")
# Récupérer les taux de change actuels
exchange_rates = get_exchange_rates_db()
if exchange_rates:
    st.sidebar.subheader("Current Exchange Rates (to GBP)")
    for currency, rate in exchange_rates.items():
        if currency in ['USD', 'EUR', 'AED', 'CAD', 'SGD']:
            st.sidebar.write(f"1 {currency} = {rate:.4f} GBP")
scenario_type = st.sidebar.selectbox(
    "Select Scenario",
    ["Current Status", "Crisis Scenario (-10%)", "Optimistic Scenario (+5%)", "Custom Scenario"]
)
if scenario_type == "Custom Scenario":
    usd_eur_change = st.sidebar.slider(
        "USD/EUR Change (%)",
        min_value=-20,
        max_value=20,
        value=0,
        step=1
    )
else:
    if scenario_type == "Crisis Scenario (-10%)":
        usd_eur_change = -10
    elif scenario_type == "Optimistic Scenario (+5%)":
        usd_eur_change = 5
    else:
        usd_eur_change = 0
# Calcul des scénarios
scenario_results = calculate_scenario(usd_eur_change)
# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💱 Exposure Analysis", "🛠️ Solutions", "📈 Scenarios"])
with tab1:
    st.header("Financial Dashboard")
   
    # Indicateurs clés
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.metric(
            "Total Exposure",
            f"£{df.loc['After Intervention', 'Total Exposure (GBP)']:,.0f}",
            delta=f"-{85}% from initial",
            delta_color="inverse"
        )
   
    with col2:
        st.metric(
            "Potential Loss",
            f"£{df.loc['After Intervention', 'Potential Loss (GBP)']:,.0f}",
            delta=f"-{85}% from initial",
            delta_color="inverse"
        )
   
    with col3:
        st.metric(
            "ROI (Year 1)",
            f"{df.loc['After Intervention', 'ROI (%)']:.0f}%",
            delta="229%"
        )
   
    with col4:
        st.metric(
            "Payback Period",
            f"{df.loc['After Intervention', 'Payback (months)']:.1f} months",
            delta="3.6 months"
        )
   
    # Graphique d'exposition
    st.subheader("Exposure Evolution")
    fig_exposure = px.bar(
        df,
        x=df.index,
        y=["Total Exposure (GBP)", "Potential Loss (GBP)"],
        barmode="group",
        title="Exposure Before and After Intervention",
        labels={"value": "Amount (GBP)", "variable": "Metric"},
        color_discrete_map={
            "Total Exposure (GBP)": "#1f77b4",
            "Potential Loss (GBP)": "#ff7f0e"
        }
    )
    st.plotly_chart(fig_exposure, use_container_width=True)
   
    # Graphique des solutions
    st.subheader("Solutions Performance")
    fig_solutions = px.bar(
        solutions_df,
        x="Solution",
        y="ROI (%)",
        color="Cost (GBP)",
        title="ROI by Solution",
        labels={"ROI (%)": "Return on Investment (%)", "Solution": "Solution"},
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_solutions, use_container_width=True)
with tab2:
    st.header("Currency Exposure Analysis")
   
    # Afficher les taux de change actuels
    if exchange_rates:
        st.subheader("Current Exchange Rates (to GBP)")
        rates_df = pd.DataFrame(list(exchange_rates.items()), columns=['Currency', 'Rate to GBP'])
        # Filtrer pour les devises pertinentes
        rates_df = rates_df[rates_df['Currency'].isin(['USD', 'EUR', 'AED', 'CAD', 'SGD'])]
        st.dataframe(rates_df, use_container_width=True)
   
    # Graphique camembert
    st.subheader("Exposure by Currency (After Intervention)")
    fig_pie = px.pie(
        st.session_state.exposure_df,
        values="Exposure After (GBP)",
        names="Currency",
        title="Currency Distribution",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)
   
    # Tableau des devises
    st.subheader("Detailed Currency Exposure")
    st.dataframe(
        st.session_state.exposure_df.style.format({
            "Exposure Before (GBP)": "{:,.0f}",
            "Exposure After (GBP)": "{:,.0f}",
            "Volatility (%)": "{:.1f}"
        }),
        use_container_width=True
    )
   
    # Graphique de volatilité
    st.subheader("Currency Volatility")
    fig_volatility = px.bar(
        st.session_state.exposure_df,
        x="Currency",
        y="Volatility (%)",
        title="Annual Volatility by Currency",
        labels={"Volatility (%)": "Volatility (%)", "Currency": "Currency"},
        color="Volatility (%)",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_volatility, use_container_width=True)
   
    # Formulaire pour mettre à jour les données d'exposition
    st.subheader("Update Exposure Data")
    with st.form("update_exposure_form"):
        st.write("Update exposure data for each currency:")
       
        updated_data = []
        for i, row in st.session_state.exposure_df.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                currency = st.text_input(f"Currency {i+1}", value=row['Currency'], key=f"currency_{i}")
            with col2:
                exposure_before = st.number_input(f"Exposure Before (GBP) {i+1}", value=row['Exposure Before (GBP)'], key=f"before_{i}")
            with col3:
                exposure_after = st.number_input(f"Exposure After (GBP) {i+1}", value=row['Exposure After (GBP)'], key=f"after_{i}")
            with col4:
                volatility = st.number_input(f"Volatility (%) {i+1}", value=row['Volatility (%)'], key=f"volatility_{i}")
           
            updated_data.append((currency, exposure_before, exposure_after, volatility))
       
        submit_button = st.form_submit_button("Update Exposure Data")
       
        if submit_button:
            # Mettre à jour le DataFrame
            st.session_state.exposure_df = pd.DataFrame(updated_data, columns=['Currency', 'Exposure Before (GBP)', 'Exposure After (GBP)', 'Volatility (%)'])
            st.success("Exposure data updated successfully!")
           
            # Si la source est SQLite, on met aussi à jour la base
            if data_source == "SQLite":
                update_exposure_data_db(updated_data)
                st.success("Data also saved to SQLite!")
with tab3:
    st.header("Implemented Solutions")
   
    # Tableau des solutions
    st.subheader("Solutions Overview")
    st.dataframe(
        solutions_df.style.format({
            "Cost (GBP)": "{:,.0f}",
            "Reduced Exposure (GBP)": "{:,.0f}",
            "Avoided Loss (GBP)": "{:,.0f}",
            "ROI (%)": "{:.0f}",
            "Timeline (days)": "{:.0f}"
        }),
        use_container_width=True
    )
   
    # Graphique des coûts vs bénéfices
    st.subheader("Cost vs Benefit Analysis")
    fig_cost_benefit = px.scatter(
        solutions_df,
        x="Cost (GBP)",
        y="Avoided Loss (GBP)",
        size="Reduced Exposure (GBP)",
        color="Solution",
        title="Cost vs Benefit of Solutions",
        labels={
            "Cost (GBP)": "Implementation Cost (GBP)",
            "Avoided Loss (GBP)": "Avoided Loss (GBP)",
            "Reduced Exposure (GBP)": "Exposure Reduced (GBP)"
        },
        size_max=60
    )
    st.plotly_chart(fig_cost_benefit, use_container_width=True)
   
    # Détails des solutions
    st.subheader("Solution Details")
    selected_solution = st.selectbox("Select a solution", solutions_df["Solution"])
   
    solution_details = solutions_df[solutions_df["Solution"] == selected_solution].iloc[0]
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.metric("Implementation Cost", f"£{solution_details['Cost (GBP)']:,.0f}")
        st.metric("Reduced Exposure", f"£{solution_details['Reduced Exposure (GBP)']:,.0f}")
   
    with col2:
        st.metric("Avoided Loss", f"£{solution_details['Avoided Loss (GBP)']:,.0f}")
        st.metric("ROI", f"{solution_details['ROI (%)']:.0f}%")
   
    st.markdown(f"**Implementation Timeline:** {solution_details['Timeline (days)']} days")
with tab4:
    st.header("Scenario Analysis")
   
    # Affichage des résultats du scénario
    st.subheader(f"Scenario: {scenario_type}")
   
    if scenario_type != "Current Status":
        col1, col2, col3 = st.columns(3)
       
        with col1:
            st.metric(
                "USD/EUR Change",
                f"{usd_eur_change}%",
                delta=None
            )
       
        with col2:
            impact_color = "inverse" if scenario_results["Total Impact (GBP)"] < 0 else "normal"
            st.metric(
                "Total Impact",
                f"£{scenario_results['Total Impact (GBP)']:,.0f}",
                delta=None,
                delta_color=impact_color
            )
       
        with col3:
            st.metric(
                "After Protection",
                f"£{scenario_results['After Protection (GBP)']:,.0f}",
                delta=f"£{scenario_results['Protection Applied (GBP)']:,.0f} protected",
                delta_color="inverse"
            )
       
        # Graphique d'impact
        st.subheader("Impact Analysis")
        scenario_df = pd.DataFrame({
            "Metric": ["Total Impact", "After Protection"],
            "Amount (GBP)": [
                scenario_results["Total Impact (GBP)"],
                scenario_results["After Protection (GBP)"]
            ]
        })
       
        fig_scenario = px.bar(
            scenario_df,
            x="Metric",
            y="Amount (GBP)",
            title="Financial Impact Analysis",
            labels={"Amount (GBP)": "Amount (GBP)", "Metric": "Metric"},
            color="Metric",
            color_discrete_map={
                "Total Impact": "#ff7f0e",
                "After Protection": "#2ca02c"
            }
        )
        st.plotly_chart(fig_scenario, use_container_width=True)
    else:
        st.info("Select a scenario to see the analysis")
# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Excis Compliance Ltd**")
st.sidebar.markdown("Currency Risk Management System")
st.sidebar.markdown(f"*Report Date: September 3, 2025*")