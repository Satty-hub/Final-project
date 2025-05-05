# Epitope Prediction App using protein_id from UniProt
# 1. Import all the required libraries here
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
import joblib
import random

# 2. upload the data for  training. I have used SARS_CoV-2 data for training.
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

# 3. Include all the peptide features to the dataset

def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['peptide_seq_length'] = df['peptide_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# 4. Predict and generate overlapping peptides of lengths 9 to 14

def generate_peptides(sequence, min_length=9, max_length=14):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

# 5. Testing the Simulate peptide dataset

def simulate_peptide_data(seq, parent_id="Protein"):
    peptides = generate_peptides(seq)
    rows = []
    for start, end, pep in peptides:
        analysis = ProteinAnalysis(pep)
        row = {
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
            "stability": round(analysis.instability_index(), 2)
        }
        rows.append(row)
    return pd.DataFrame(rows)

# 6. Put your protein_d and fetch sequence from UniProt

def fetch_protein_sequence_uniprot(protein_id):
    url = f"https://www.uniprot.org/uniprot/{protein_id}.fasta"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    fasta_io = StringIO(response.text)
    record = SeqIO.read(fasta_io, "fasta")
    return str(record.seq)

# 7. Once you define all the feature and source of data to fetch then set the code for Streamlit 

st.set_page_config(layout="wide")
st.title("🔬 Epitope Prediction from Protein Sequences")
page = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Epitope Prediction"])
df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

if page == "Data Overview":
    st.header("📊 Data Overview")
    st.dataframe(df_bcell.head())

elif page == "Model Training":
    st.header("🤖 Train Your Model")
    choice = st.selectbox("Select Dataset", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t
    df = add_features(df)

    FEATURE_COLUMNS = [
        'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
        'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
        'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
    ]

    df = df.dropna(subset=['target'])
    X = df[FEATURE_COLUMNS]
    Y = df['target']

    if st.checkbox("Apply SMOTE"):
        smote = SMOTE()
        X, Y = smote.fit_resample(X, Y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    if st.button("Train Model"):
        model = RandomForestClassifier(n_estimators=500, random_state=42)
        model.fit(X_train, Y_train)
        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{choice.lower()}-scaler.pkl")
        st.success("Model trained and saved.")

elif page == "Epitope Prediction":
    st.header("🔎 Predict B-cell Epitopes")
    protein_id = st.text_input("Enter UniProt Accession ID (e.g. P0DTC2):")

    sequence = ""
    if protein_id:
        sequence = fetch_protein_sequence_uniprot(protein_id)
        if sequence:
            st.success("Sequence retrieved from UniProt.")
        else:
            st.error("Could not retrieve sequence. Check ID.")

    sequence = st.text_area("Or paste sequence here:", value=sequence, height=150)

    if st.button("Predict Epitopes"):
        if sequence:
            df = simulate_peptide_data(sequence)
            df_features = add_features(df)
            feature_cols = [
                'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
                'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
                'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
            ]
            X_pred = df_features[feature_cols]

            try:
                model = joblib.load("b-cell-rf_model.pkl")
                scaler = joblib.load("b-cell-scaler.pkl")
                X_scaled = scaler.transform(X_pred)
                preds = model.predict(X_scaled)

                df_features['prediction'] = preds
                st.dataframe(df_features[df_features['prediction'] == 1])

                st.download_button("Download Predictions", data=df_features.to_csv(index=False), file_name="predicted_epitopes.csv")
            except Exception as e:
                st.error(f"Model loading or prediction error: {e}")
