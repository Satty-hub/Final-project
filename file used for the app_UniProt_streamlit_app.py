# Epitope prediction tool using streamlit and Python

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

# Background and navigator Config and Custom Styling 

import streamlit as st

# Set page configuration
st.set_page_config(layout="wide", page_title="Epitope Predictor")

# Sidebar with image and custom CSS
st.markdown("""
    <style>
        /* Main app background styling */
        .stApp {
            background-image: 
                url("https://images.unsplash.com/photo-1583324113626-70df0f4deaab?auto=format&fit=crop&w=2100&q=80"),
                url("https://images.unsplash.com/photo-1501605144124-f7f1cf515d16?auto=format&fit=crop&w=500&h=500&dpr=2");
            background-size: cover, contain;
            background-attachment: fixed;
            background-repeat: no-repeat, no-repeat;
            background-position: center, bottom right;
        }

        /* Sidebar background styling (left side only) */
        section[data-testid="stSidebar"] {
            position: relative;
            background-color: rgba(255, 255, 255, 0.9); /* Slight transparency for sidebar */
            height: 100vh;  /* Full height for the sidebar */
            padding-top: 60px; /* Adds space for the Navigator Menu */
            width: 250px;  /* Sidebar width */
        }

        /* Adding the image background only below the Navigator menu */
        section[data-testid="stSidebar"]::after {
            content: "";
            position: absolute;
            top: 60px;  /* Position the image below the Navigator menu */
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('https://media.healthdirect.org.au/images/inline/original/organs-of-the-immune-system-illustration-18584a.jpg');
            background-size: contain; /* Ensure image is contained within the sidebar */
            background-repeat: no-repeat;
            background-position: center;
            z-index: -1; /* Keeps the image behind the sidebar content */
        }

        /* Content block styling */
        .block-container {
            background-color: rgba(255, 255, 255, 0.85); /* Slight transparency */
            padding: 2rem;
            border-radius: 1rem;
            margin-top: 2rem;
        }

        /* Text styles for headers */
        h1, h2, h3 {
            color: #1e3d59;
        }

    </style>
""", unsafe_allow_html=True)

# Step 0.1: Define UniProt fetch function
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

# Step 1: Upload the dataset
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

# Step 2: Add feature engineering
def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# Step 3: Peptide generation
def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

# Step 4: Simulate peptide data
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
        except:
            continue
    return pd.DataFrame(rows)

# Step 5: Streamlit App Layout
st.title("B-cell and T-cell Epitope Predictor")
page = st.sidebar.radio("Navigation", ["Model Training", "T cell epitope predictor", "B cell epitope predictor", "Data Overview"])
df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

# Step 6: Data Overview Page
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

# Step 7: Model Training Page
elif page == "Model Training":
    st.header("Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t
    df = add_features(df)

    FEATURE_COLUMNS = [
        'protein_seq_length', 'parent_protein_id_length', 'peptide_length',
        'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker',
        'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
    ]

    df = df.dropna(subset=['target'])
    df = df.drop(columns=[col for col in ["parent_protein_id", "protein_seq", "peptide_seq"] if col in df.columns])

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    X = df[FEATURE_COLUMNS]
    Y = df["target"]

    if st.checkbox("Apply SMOTE"):
        sm = SMOTE()
        X, Y = sm.fit_resample(X, Y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    test_size = st.slider("Test Size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random Seed", 0, 100, 42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    if st.button("Train Model"):
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        st.success("Model trained!")
        st.write("Accuracy:", accuracy_score(Y_test, Y_pred))
        st.text(classification_report(Y_test, Y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{choice.lower()}-scaler.pkl")

# Step 8: Epitope Predictor (T-cell & B-cell)
elif page in ["T cell epitope predictor", "B cell epitope predictor"]:
    st.header("Epitope Prediction")
    model_type = "T-cell" if "T" in page else "B-cell"

    organism = st.selectbox("Select Organism", ["Human", "Virus", "Bacteria", "Fungi", "Mouse", "Other"])
    uniprot_id = st.text_input("Enter UniProt ID (optional)")
    sequence = st.text_area("Paste Protein Sequence:", height=200)
    protein_name = st.text_input("Protein Name", "Manual_Protein")

    if uniprot_id:
        sequence, protein_name = fetch_sequence_from_uniprot(uniprot_id)
        if not sequence:
            st.warning("Could not fetch sequence from UniProt. Please paste it manually below.")

    if st.button("Generate & Predict") and sequence.strip():
        df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)
        df = add_features(df)

        model_file = f"{model_type.lower()}-rf_model.pkl"
        scaler_file = f"{model_type.lower()}-scaler.pkl"

        if not os.path.exists(model_file):
            st.error("Train model first.")
            st.stop()

        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)

        feature_cols = [
            'protein_seq_length', 'parent_protein_id_length', 'peptide_length',
            'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker',
            'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
        ]

        if not all(col in df.columns for col in feature_cols):
            st.error("Some features are missing.")
            st.stop()

        X_pred = scaler.transform(df[feature_cols])
        df['prediction'] = model.predict(X_pred)

        st.dataframe(df)

        st.subheader("Download Predicted Peptides")
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{protein_name}_predictions.csv")

        st.subheader("Immunogenicity Score Distribution")
        st.plotly_chart(px.box(df, y="immunogenicity_score", title="Immunogenicity Score Distribution"))

        st.subheader("Feature Distribution by Prediction")

        st.subheader("Feature Relationships with Immunogenicity Score")

        for feature in feature_cols:
              fig = px.scatter(df, x=feature, y="immunogenicity_score", color="prediction", trendline="ols",
                     title=f"{feature} vs Immunogenicity Score")
              st.plotly_chart(fig)

        st.subheader("Stability Distribution")
        st.plotly_chart(px.box(df, y="stability"))

        st.subheader("Peptide Length Distribution")
        st.plotly_chart(px.histogram(df, x="peptide_length"))

        st.subheader("Aromaticity Distribution")
        st.plotly_chart(px.violin(df, y="aromaticity", box=True))

        st.subheader("📍 Start and End Position Distributions")

       # Start position distribution
        fig_start = px.violin(df, y="start_position", box=True, points="all", color="prediction", title="Start Position Distribution")
        st.plotly_chart(fig_start)

      # End position distribution
        fig_end = px.violin(df, y="end_position", box=True, points="all", color="prediction", title="End Position Distribution")
        st.plotly_chart(fig_end)
