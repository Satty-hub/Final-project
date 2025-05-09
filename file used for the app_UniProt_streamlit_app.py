# This Python (Pandas) code can be used to predict the T-cell and B-cell epitope using UniProt ID or Protein sequence

# Epitope Predictor App
# ---------------------
# Step 0: Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import requests
import random
import joblib
import os

# ---------------------
# Step 1: Streamlit page configuration and UI Styling
st.set_page_config(page_title="Epitope Predictor", layout="wide")

with st.sidebar:
    st.image("https://i0.wp.com/immunodeficiency.ca/wp-content/uploads/2012/05/Immune-System-Diagram-en.jpg?ssl=1", caption="Immune System")
    st.markdown("### Navigation")

# ---------------------
# Step 2: Load datasets
@st.cache_data
def load_data():
    bcell_url = "https://drive.google.com/uc?id=1_v_AiVvwpSnuKCNplAimFS8sOlu-hZeQ&export=download"
    covid_url = "https://drive.google.com/uc?id=13JRk-wG8GggBTA-3J1U4R5x3nhrT7KbY&export=download"
    sars_url = "https://drive.google.com/uc?id=1hlza1PsXkHiBqzhpZpVKcLDlLUs4aQtj&export=download"
    Tcell_url = "https://drive.google.com/uc?id=1wYhEDx7pRxiHzD58R2ihfDrSp5Bu68cc&export=download"

    df_bcell = pd.read_csv(bcell_url)
    df_tcell = pd.read_csv(Tcell_url)
    df_sars = pd.read_csv(sars_url)
    df_test = pd.read_csv(covid_url)

    df_train_b = pd.concat([df_bcell, df_sars])
    df_train_t = pd.concat([df_tcell, df_sars])

    return df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t

# ---------------------
# Step 3: Feature engineering
def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# ---------------------
# Step 4: Peptide generator
def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

# ---------------------
# Step 5: Peptide data simulator
def simulate_peptide_data(seq, parent_id="Unknown", organism="Unknown"):
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    peptides = generate_peptides(seq)
    rows = []

    for start, end, pep in peptides:
        if not set(pep).issubset(valid_aa):
            continue
        try:
            analysis = ProteinAnalysis(pep)
            row = {
                "organism": organism,
                "parent_protein_id": parent_id,
                "protein_seq": seq,
                "start_position": start,
                "end_position": end,
                "peptide_seq": pep,
                "chou_fasman": round(random.uniform(0.2, 1.0), 3),
                "emini": round(random.uniform(0.5, 2.5), 3),
                "kolaskar_tongaonkar": round(random.uniform(0.8, 1.2), 3),
                "parker": round(random.uniform(0.5, 3.0), 3),
                "isoelectric_point": round(analysis.isoelectric_point(), 2),
                "aromaticity": round(analysis.aromaticity(), 3),
                "hydrophobicity": round(analysis.gravy(), 3),
                "stability": round(analysis.instability_index(), 2),
                "immunogenicity_score": round(random.uniform(0.0, 1.0), 3)
            }
            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)

# ---------------------
# Step 6: UniProt fetcher
def fetch_sequence_from_uniprot(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    try:
        response = requests.get(url)
        if response.ok:
            lines = response.text.strip().split('\n')
            sequence = ''.join(lines[1:])
            name = lines[0].split('|')[2] if '|' in lines[0] else "Protein"
            return sequence, name
    except Exception:
        pass
    return "", ""

# ---------------------
# Step 7: CSV exporter
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------------------
# Step 8: Main Navigation
df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

page = st.sidebar.radio("Go to", ["Data Overview", "Model Training", "T cell epitope predictor", "B cell epitope predictor"])

# ---------------------
# Step 9: Data Overview
if page == "Data Overview":
    st.header("Step 1: Data Overview")
    st.subheader("B-cell Data")
    st.dataframe(df_bcell.head())
    st.subheader("T-cell Data")
    st.dataframe(df_tcell.head())

# ---------------------
# Step 10: Model Training
elif page == "Model Training":
    st.header("Step 2: Train Epitope Prediction Model")

    model_type = st.selectbox("Choose model type to train", ["B-cell", "T-cell"])
    df_train = df_train_b if model_type == "B-cell" else df_train_t

    df_train = add_features(df_train)
    df_train = df_train.dropna(subset=["target"])

    X = df_train[['protein_seq_length', 'parent_protein_id_length', 'peptide_length', 'chou_fasman',
                  'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point',
                  'aromaticity', 'hydrophobicity', 'stability']]
    y = df_train["target"]

    if st.checkbox("Apply SMOTE"):
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)
        st.success("SMOTE applied")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    test_size = st.slider("Test size", 0.1, 0.5, 0.3)
    seed = st.number_input("Random seed", 0, 100, 42)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=seed)

    if st.button("Train Model"):
        model = RandomForestClassifier(n_estimators=300, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        joblib.dump(model, f"{model_type.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{model_type.lower()}-scaler.pkl")
        st.success("Model & Scaler saved!")

# ---------------------
# Step 11: T-cell / B-cell Epitope Prediction
elif page in ["T cell epitope predictor", "B cell epitope predictor"]:
    st.header("Step 3: Predict Epitopes")

    model_type = "T-cell" if page == "T cell epitope predictor" else "B-cell"
    organism = st.selectbox("Organism", ["Human", "Virus", "Bacteria", "Other"])
    uniprot_id = st.text_input("Enter UniProt ID (optional)")
    sequence = ""
    protein_name = ""

    if uniprot_id:
        sequence, protein_name = fetch_sequence_from_uniprot(uniprot_id)

    if not sequence:
        sequence = st.text_area("Paste Protein Sequence:")
        protein_name = st.text_input("Protein Name", "Manual_Protein")

    if sequence.strip() and st.button("Predict Epitopes"):
        try:
            df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)
            df_features = add_features(df)

            feature_cols = ['protein_seq_length', 'parent_protein_id_length', 'peptide_length',
                            'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker',
                            'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability']

            model_file = f"{model_type.lower()}-rf_model.pkl"
            scaler_file = f"{model_type.lower()}-scaler.pkl"

            if not os.path.exists(model_file):
                st.error("Model not found. Please train the model first.")
            else:
                model = joblib.load(model_file)
                scaler = joblib.load(scaler_file)
                X_pred = df_features[feature_cols]
                X_scaled = scaler.transform(X_pred)

                df_features['prediction'] = model.predict(X_scaled)

                st.dataframe(df_features)

                fig1 = px.violin(df_features, y="immunogenicity_score", box=True, title="Immunogenicity Score")
                st.plotly_chart(fig1)

                fig2 = px.box(df_features, y="hydrophobicity", title="Hydrophobicity")
                st.plotly_chart(fig2)

                fig3, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(df_features[feature_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig3)

                csv = convert_df_to_csv(df_features)
                st.download_button("Download Predictions", csv, file_name="epitope_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
