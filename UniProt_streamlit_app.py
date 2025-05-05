# This Python (Pandas) code can be used to predict the Tcell and B cell epitoe using Uni_prot ID or Protein sequence
# # Import all required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import requests
import random
import joblib

# Step 1: Upload the data
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

# Step 2: Feature Engineering

def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['peptide_seq_length'] = df['peptide_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# Step 3: Generate peptides

def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

# Step 4: Simulate data with features

def simulate_peptide_data(seq, parent_id="Unknown", organism="Unknown"):
    peptides = generate_peptides(seq)
    rows = []
    for start, end, pep in peptides:
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

# Step 5: Fetch sequence from UniProt

def fetch_sequence_from_uniprot(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.ok:
        fasta = response.text
        lines = fasta.split("\n")
        seq = "".join(lines[1:])
        name = lines[0].split("|")[-1].strip()
        return seq, name
    return None, None

# Step 6: Streamlit UI

st.set_page_config(layout="wide")
st.title("🔬 B-cell and T-cell Epitope Predictor")

page = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Epitope Prediction"])
df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

if page == "Data Overview":
    st.header("📊 Data Overview")
    st.dataframe(df_bcell.head())
    st.dataframe(df_tcell.head())
    st.dataframe(df_sars.head())

if page == "Model Training":
    st.header("🤖 Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t
    df = add_features(df)
    FEATURE_COLUMNS = [
        'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
        'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
        'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
    ]
    df = df.dropna(subset=['target'])
    X = df[FEATURE_COLUMNS]
    Y = df["target"]
    if st.checkbox("Apply SMOTE"):
        smote = SMOTE()
        X, Y = smote.fit_resample(X, Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    if st.button("Train Model"):
        model = RandomForestClassifier(n_estimators=500)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(Y_test, Y_pred))
        st.text(classification_report(Y_test, Y_pred))
        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{choice.lower()}-scaler.pkl")

if page == "Epitope Prediction":
    st.header("🔎 Epitope Prediction")
    organism = st.selectbox("Organism", ["Human", "Virus", "Bacteria"])
    uniprot_id = st.text_input("UniProt ID")
    default_seq = "MFVFLVLLPLVSSQCVNL..."
    sequence = None
    protein_name = "Unknown"
    if uniprot_id:
        sequence, protein_name = fetch_sequence_from_uniprot(uniprot_id)
    if not sequence:
        sequence = st.text_area("Paste Protein Sequence", default_seq)
        protein_name = st.text_input("Protein Name", "Manual_Protein")

    if st.button("Compare B-cell and T-cell Predictions"):
        full_results = {}
        for model_type in ["B-cell", "T-cell"]:
            df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)
            df_features = add_features(df)
            feature_cols = [
                'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
                'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
                'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
            ]
            model = joblib.load(f"{model_type.lower()}-rf_model.pkl")
            scaler = joblib.load(f"{model_type.lower()}-scaler.pkl")
            X_pred = scaler.transform(df_features[feature_cols])
            preds = model.predict(X_pred)
            df_features['prediction'] = preds
            df_features['model_type'] = model_type
            full_results[model_type] = df_features

        b_df = full_results["B-cell"]
        t_df = full_results["T-cell"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("B-cell Prediction Summary")
            b_pos = b_df[b_df['prediction'] == 1]
            st.metric("Positive Epitopes", len(b_pos))
            st.metric("Avg Length", round(b_pos['peptide_length'].mean(), 2))
            st.metric("Total Length", round(b_pos['peptide_length'].sum(), 2))
            st.plotly_chart(px.histogram(b_pos, x='peptide_length', title='B-cell Length'))
        with col2:
            st.subheader("T-cell Prediction Summary")
            t_pos = t_df[t_df['prediction'] == 1]
            st.metric("Positive Epitopes", len(t_pos))
            st.metric("Avg Length", round(t_pos['peptide_length'].mean(), 2))
            st.metric("Total Length", round(t_pos['peptide_length'].sum(), 2))
            st.plotly_chart(px.histogram(t_pos, x='peptide_length', title='T-cell Length'))

        st.subheader("📥 Download Combined Prediction CSV")
        combined = pd.concat([b_df, t_df])
        csv = combined.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("Download CSV", data=csv.encode('utf-8-sig'), file_name="combined_predictions.csv")

