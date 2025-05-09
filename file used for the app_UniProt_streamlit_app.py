# This Python (Pandas) code can be used to predict the T-cell and B-cell epitope using UniProt ID or Protein sequence

# Epitope Predictor App with Background, Model Training, and Enhanced Visualizations

# -----------------------------
# Step 0: Import Required Libraries
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# -----------------------------
# Step 1: Set Background Image
# -----------------------------
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('{image_url}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://upload.wikimedia.org/wikipedia/commons/6/6a/SARS-CoV-2_without_background.png")

# -----------------------------
# Step 2: Load Data
# -----------------------------
@st.cache_data

def load_data():
    bcell_url = "https://drive.google.com/uc?id=1_v_AiVvwpSnuKCNplAimFS8sOlu-hZeQ&export=download"
    covid_url = "https://drive.google.com/uc?id=13JRk-wG8GggBTA-3J1U4R5x3nhrT7KbY&export=download"
    sars_url = "https://drive.google.com/uc?id=1hlza1PsXkHiBqzhpZpVKcLDlLUs4aQtj&export=download"
    tcell_url = "https://drive.google.com/uc?id=1wYhEDx7pRxiHzD58R2ihfDrSp5Bu68cc&export=download"

    df_bcell = pd.read_csv(bcell_url)
    df_tcell = pd.read_csv(tcell_url)
    df_sars = pd.read_csv(sars_url)
    df_test = pd.read_csv(covid_url)

    df_train_b = pd.concat([df_bcell, df_sars])
    df_train_t = pd.concat([df_tcell, df_sars])
    return df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t

df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

# -----------------------------
# Step 3: Utility Functions
# -----------------------------
def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

def simulate_peptide_data(seq, parent_id="Unknown", organism="Unknown"):
    peptides = generate_peptides(seq)
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    rows = []
    for start, end, pep in peptides:
        if not set(pep).issubset(valid_aa):
            continue
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
    return pd.DataFrame(rows)

def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# -----------------------------
# Step 4: App Layout and Logic
# -----------------------------
st.title("B-cell and T-cell Epitope Predictor")
page = st.sidebar.radio("Navigation", ["Model Training", "T cell epitope predictor", "B cell epitope predictor", "Data Overview"])

# -----------------------------
# Step 5: Data Overview
# -----------------------------
if page == "Data Overview":
    st.header("Data Overview")
    st.subheader("B-cell Dataset")
    st.dataframe(df_bcell.head())
    st.subheader("T-cell Dataset")
    st.dataframe(df_tcell.head())
    st.subheader("SARS Dataset")
    st.dataframe(df_sars.head())
    st.subheader("COVID Test Dataset")
    st.dataframe(df_test.head())

# -----------------------------
# Step 6: Model Training
# -----------------------------
elif page == "Model Training":
    st.header("Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t

    df = df.drop(columns=[col for col in ["parent_protein_id", "protein_seq", "peptide_seq", "start_position", "end_position"] if col in df.columns])
    if "target" not in df.columns:
        st.error("Missing 'target' column in the dataset.")
        st.stop()

    df = df.dropna(subset=['target'])
    feature_cols = [
        'protein_seq_length', 'parent_protein_id_length', 'peptide_length',
        'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker',
        'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
    ]

    X = df[feature_cols]
    Y = df['target']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if st.checkbox("Apply SMOTE for balancing"):
        smote = SMOTE()
        X, Y = smote.fit_resample(X, Y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", 0, 100, 42)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    if st.button("Train Random Forest"):
        model = RandomForestClassifier(n_estimators=500, random_state=random_state)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(Y_test, Y_pred))
        st.text(classification_report(Y_test, Y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{choice.lower()}-scaler.pkl")

# -----------------------------
# Step 7: Epitope Predictor
# -----------------------------
elif page in ["T cell epitope predictor", "B cell epitope predictor"]:
    st.header("Epitope Predictor")
    model_type = "T-cell" if "T" in page else "B-cell"

    organism = st.selectbox("Select Organism", ["Human", "Bacteria", "Virus", "Fungi", "Mice", "Other"])
    sequence = st.text_area("Paste Protein Sequence:", height=150)
    protein_name = st.text_input("Protein Name", "Manual_Protein")

    if st.button("Generate Epitopes and Predict"):
        try:
            df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)
            df_features = add_features(df)
            feature_cols = [
                'protein_seq_length', 'parent_protein_id_length', 'peptide_length',
                'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker',
                'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
            ]
            model_file = f"{model_type.lower()}-rf_model.pkl"
            scaler_file = f"{model_type.lower()}-scaler.pkl"

            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                st.error("Model or scaler not found. Please train first.")
                st.stop()

            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            X_pred = scaler.transform(df_features[feature_cols])
            df_features['prediction'] = model.predict(X_pred)

            st.dataframe(df_features)

            # Plots
            st.subheader("Plots")
            st.plotly_chart(px.histogram(df_features, x="stability", color="prediction", title="Stability Distribution"))
            st.plotly_chart(px.box(df_features, y="peptide_length", color="prediction", title="Peptide Length Distribution"))
            st.plotly_chart(px.violin(df_features, y="aromaticity", box=True, points="all", color="prediction", title="Aromaticity Violin Plot"))
            st.plotly_chart(px.scatter(df_features, x="start_position", y="end_position", color="prediction", title="Start vs End Position"))

            csv = convert_df_to_csv(df_features)
            st.download_button("Download Predictions", csv, f"{protein_name}_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {str(e)}")
