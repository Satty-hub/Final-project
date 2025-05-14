# Epitope prediction tool using Streamlit and Python

# Step 0: Imports
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import joblib
import random
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Step 0.1: Page config and background
st.set_page_config(layout="wide", page_title="Epitope Predictor")

# Step 0.2: Background CSS
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1583324113626-70df0f4deaab?auto=format&fit=crop&w=2100&q=80");
            background-size: cover;
            background-attachment: fixed;
        }
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.9);
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 1rem;
            margin-top: 2rem;
        }
        h1, h2, h3 {
            color: #1e3d59;
        }
    </style>
""", unsafe_allow_html=True)

# Step 1: Fetch UniProt Sequence
def fetch_sequence_from_uniprot(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text.strip().split('\n')
        protein_name = fasta_data[0][1:]
        sequence = ''.join(fasta_data[1:])
        return sequence, protein_name
    else:
        return None, None

# Step 2: Generate peptides
def generate_peptides(sequence, peptide_length):
    return [sequence[i:i+peptide_length] for i in range(len(sequence) - peptide_length + 1)]

# Step 3: MHC Class I Prediction using mhcflurry
def predict_mhc_class_i(peptides, allele):
    predictor = Class1AffinityPredictor.load()
    df = predictor.predict_to_dataframe(peptides=peptides, allele=allele)
    df['is_epitope'] = df['affinity'] < 500
    return df

# Step 4: MHC Class II Prediction using netMHCIIpan
def predict_mhc_class_ii(peptides, allele):
    results = []
    for pep in peptides:
        try:
            output = subprocess.check_output(
                ['netMHCIIpan', '-a', allele, '-f', '-', '-length', str(len(pep))],
                input=pep.encode(),
                stderr=subprocess.DEVNULL
            ).decode()
            for line in output.split('\n'):
                if line.startswith(allele):
                    fields = line.split()
                    affinity = float(fields[-1])
                    results.append({"peptide": pep, "affinity": affinity, "is_epitope": affinity < 1000})
        except:
            continue
    return pd.DataFrame(results)

# Step 5: Dummy B-cell and T-cell prediction

def predict_b_cell_epitopes(sequence):
    peptides = generate_peptides(sequence, 15)
    return pd.DataFrame({"Peptide": peptides, "Score": np.random.rand(len(peptides))})

def predict_t_cell_epitopes(sequence):
    peptides = generate_peptides(sequence, 9)
    return pd.DataFrame({"Peptide": peptides, "Score": np.random.rand(len(peptides))})

# Step 6: Sidebar & Navigation
st.sidebar.title("Epitope Prediction App")
page = st.sidebar.radio("Go to", [
    "Home",
    "B-cell Prediction",
    "T-cell Prediction",
    "MHC Class I Predictor",
    "MHC Class II Predictor"
])

# Step 7: Page content
if page == "Home":
    st.title("Welcome to the Epitope Prediction App")
    st.write("""
        This tool allows you to:
        - Predict B-cell and T-cell epitopes
        - Predict MHC Class I and II epitopes using local tools
        - Input your own sequence or fetch it from UniProt
        - Customize peptide length and allele type
    """)
    st.info("Choose an option from the sidebar to get started.")

elif page == "B-cell Prediction":
    st.title("B-cell Epitope Prediction")
    sequence = st.text_area("Paste protein sequence here", height=200)
    uniprot_id = st.text_input("Or enter a UniProt ID")

    if uniprot_id:
        sequence, name = fetch_sequence_from_uniprot(uniprot_id)
        if sequence:
            st.success(f"Fetched sequence: {name}")
        else:
            st.error("Failed to fetch sequence from UniProt.")

    if sequence:
        if st.button("Predict B-cell Epitopes"):
            df = predict_b_cell_epitopes(sequence)
            st.dataframe(df)
            st.download_button("Download Results", df.to_csv(index=False),
                               file_name="bcell_epitope_results.csv")

elif page == "T-cell Prediction":
    st.title("T-cell Epitope Prediction")
    sequence = st.text_area("Paste protein sequence here", height=200)
    uniprot_id = st.text_input("Or enter a UniProt ID")

    if uniprot_id:
        sequence, name = fetch_sequence_from_uniprot(uniprot_id)
        if sequence:
            st.success(f"Fetched sequence: {name}")
        else:
            st.error("Failed to fetch sequence from UniProt.")

    if sequence:
        if st.button("Predict T-cell Epitopes"):
            df = predict_t_cell_epitopes(sequence)
            st.dataframe(df)
            st.download_button("Download Results", df.to_csv(index=False),
                               file_name="tcell_epitope_results.csv")

elif page == "MHC Class I Predictor":
    st.title("MHC Class I Epitope Prediction")
    sequence = st.text_area("Paste protein sequence here", height=200)
    uniprot_id = st.text_input("Or enter a UniProt ID")

    if uniprot_id:
        sequence, name = fetch_sequence_from_uniprot(uniprot_id)
        if sequence:
            st.success(f"Fetched sequence: {name}")
        else:
            st.error("Failed to fetch sequence from UniProt.")

    if sequence:
        peptide_length = st.slider("Select peptide length", 8, 15, 9)
        allele = st.selectbox("Select MHC Class I allele", [
            "HLA-A*02:01", "HLA-A*03:01", "HLA-B*07:02"
        ])
        if st.button("Predict"):
            peptides = generate_peptides(sequence, peptide_length)
            df_pred = predict_mhc_class_i(peptides, allele)
            st.dataframe(df_pred)
            st.download_button("Download Results", df_pred.to_csv(index=False),
                               file_name="mhc_class_i_results.csv")

elif page == "MHC Class II Predictor":
    st.title("MHC Class II Epitope Prediction")
    sequence = st.text_area("Paste protein sequence here", height=200)
    uniprot_id = st.text_input("Or enter a UniProt ID")

    if uniprot_id:
        sequence, name = fetch_sequence_from_uniprot(uniprot_id)
        if sequence:
            st.success(f"Fetched sequence: {name}")
        else:
            st.error("Failed to fetch sequence from UniProt.")

    if sequence:
        peptide_length = st.slider("Select peptide length", 12, 25, 15)
        allele = st.selectbox("Select MHC Class II allele", [
            "HLA-DRB1*01:01", "HLA-DRB1*04:01"
        ])
        if st.button("Predict"):
            peptides = generate_peptides(sequence, peptide_length)
            df_pred = predict_mhc_class_ii(peptides, allele)
            if not df_pred.empty:
                st.dataframe(df_pred)
                st.download_button("Download Results", df_pred.to_csv(index=False),
                                   file_name="mhc_class_ii_results.csv")
            else:
                st.warning("No predictions were made. Check your input or tool availability.")
