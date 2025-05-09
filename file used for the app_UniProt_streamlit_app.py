# This Python (Pandas) code can be used to predict the T-cell and B-cell epitope using UniProt ID or Protein sequence

# Step 1: Import all required libraries
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

# Step 2: Set page config and styling
st.set_page_config(layout="wide", page_title="Epitope Predictor")

# Sidebar image and navigation
with st.sidebar:
    st.image("https://i0.wp.com/immunodeficiency.ca/wp-content/uploads/2012/05/Immune-System-Diagram-en.jpg?ssl=1", caption="Human Immune System Diagram", use_container_width=True)  
    page = st.radio("Navigation", ["Model Training", "T cell epitope predictor", "B cell epitope predictor", "Data Overview"])

# Styling
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1583324113626-70df0f4deaab?auto=format&fit=crop&w=2100&q=80");
            background-size: cover;
            background-attachment: fixed;
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Step 3: Load data from sources
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

def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

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

def fetch_sequence_from_uniprot(uniprot_id):
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        response = requests.get(url)
        if response.ok:
            lines = response.text.split('\n')
            name = lines[0].split("|")[2] if '|' in lines[0] else "Protein"
            sequence = ''.join(lines[1:])
            return sequence, name
    except:
        pass
    return None, None

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Step 4: Load the datasets

df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

# Step 5: Navigation Handling
if page == "Data Overview":
    st.header("Data Overview")
    st.dataframe(df_bcell.head())
    st.dataframe(df_tcell.head())
    st.dataframe(df_sars.head())
    st.dataframe(df_test.head())

elif page == "Model Training":
    st.header("Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t

    feature_cols = ['protein_seq_length', 'parent_protein_id_length', 'peptide_length', 'chou_fasman',
                    'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point', 'aromaticity', 
                    'hydrophobicity', 'stability']

    df = add_features(df).dropna(subset=['target'])
    X = df[feature_cols]
    Y = df['target']
    X = StandardScaler().fit_transform(X)

    if st.checkbox("Apply SMOTE for balancing"):
        X, Y = SMOTE().fit_resample(X, Y)
        st.success("SMOTE applied")

    X = MinMaxScaler().fit_transform(X)
    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", 0, 100, 42)

    if st.button("Train Random Forest"):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        model = RandomForestClassifier(n_estimators=500, random_state=random_state)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(Y_test, Y_pred))
        st.text(classification_report(Y_test, Y_pred))
        sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='d')
        st.pyplot(plt.gcf())
        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(MinMaxScaler(), f"{choice.lower()}-scaler.pkl")

elif page in ["T cell epitope predictor", "B cell epitope predictor"]:
    st.header(f"{page}")
    organism = st.selectbox("Organism", ["Human", "Virus", "Bacteria", "Fungi", "Mice", "Other"])
    uniprot_id = st.text_input("UniProt ID (Optional)")
    sequence, protein_name = fetch_sequence_from_uniprot(uniprot_id) if uniprot_id else (None, None)

    if not sequence:
        sequence = st.text_area("Paste Protein Sequence:")
        protein_name = st.text_input("Protein Name", value="Manual_Protein")

    if st.button("Generate Epitopes and Predict") and sequence.strip():
        model_type = "tcell" if "T cell" in page else "bcell"
        model_file = f"{model_type}-rf_model.pkl"
        scaler_file = f"{model_type}-scaler.pkl"
        if not os.path.exists(model_file):
            st.error("Model not found. Please train it first.")
        else:
            df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)
            df = add_features(df)
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            X = scaler.transform(df[feature_cols])
            df['prediction'] = model.predict(X)
            st.dataframe(df)
            st.plotly_chart(px.violin(df, y="immunogenicity_score", box=True))
            st.plotly_chart(px.box(df, y="hydrophobicity"))
            sns.heatmap(df[feature_cols].corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt.gcf())
            st.download_button("Download CSV", convert_df_to_csv(df), f"{protein_name}_epitopes.csv")

        except Exception as e:
            st.error(f"Error in prediction or visualization: {str(e)}")
