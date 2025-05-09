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
st.set_page_config(layout="wide", page_title="Epitope Predictor")

# Sidebar with image of the human immune system
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="Placeholder Image", use_container_width=True)  
    st.markdown("<br>", unsafe_allow_html=True)  # Adding some space for styling
    st.markdown("""
        <style>
            .stSidebar {
                background-color: rgba(0, 0, 0, 0.1);  /* Adding light background for contrast */
            }
        </style>
    """, unsafe_allow_html=True)

# Add custom CSS background and style
st.markdown("""
    <style>
        .stApp {
            background-image: 
                url("https://images.unsplash.com/photo-1583324113626-70df0f4deaab?auto=format&fit=crop&w=2100&q=80"),
                url("https://images.unsplash.com/photo-1501605144124-f7f1cf515d16?auto=format&fit=crop&w=500&h=500&dpr=2");
            background-size: cover, contain;
            background-attachment: fixed;
            background-repeat: no-repeat, no-repeat;
            background-position: center, bottom right;
        }

        /* Stylish navigation at top-left */
        .stRadio {
            position: absolute;
            top: 20px;
            left: 20px;
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            padding: 15px 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            font-size: 16px;
            font-weight: bold;
            color: #333;
            transition: background-color 0.3s ease;
        }

        .stRadio label {
            font-size: 18px;
            color: #1e3d59;
        }

        .stRadio div.stRadioItem {
            padding: 10px;
            margin: 5px 0;
        }

        .stRadio div.stRadioItem:hover {
            background-color: rgba(0, 123, 255, 0.1);
            cursor: pointer;
        }

        /* Block container styling */
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

# Step 1: Upload the dataset, since data is in download I use the link for download

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

# ---------- Step 2: Feature Engineering ----------

def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# ---------- Step 3: Peptide Generation ----------

def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

# ---------- Step 4: Simulate Peptide Feature Data ----------

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
        except Exception as e:
            st.error(f"Error generating peptide: {e}")
            continue
    df = pd.DataFrame(rows)
    if 'immunogenicity_score' not in df.columns:
        st.error("immunogenicity_score column is missing in peptide data!")
    return df

# ---------- Step 5: Streamlit Layout ----------

st.title("B-cell and T-cell Epitope Predictor")
page = st.sidebar.radio("Navigation", ["Model Training", "T cell epitope predictor", "B cell epitope predictor", "Data Overview"])
df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

# ---------- Step 6: Data Overview ----------
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

# ---------- Step 7: Model Training ----------
elif page == "Model Training":
    st.header("Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t
    df = add_features(df)

    FEATURE_COLUMNS = [
        'protein_seq_length', 'parent_protein_id_length', 'peptide_length',
        'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker',
        'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability', 'immunogenicity_score'
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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Save the model and scaler for prediction use
    if st.button("Save Model"):
        model_filename = f"{choice.lower()}-rf_model.pkl"
        scaler_filename = f"{choice.lower()}-scaler.pkl"
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        st.success(f"Model and Scaler saved as {model_filename} and {scaler_filename}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {acc * 100:.2f}%")
    st.write(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    st.pyplot(fig)

# ---------- Step 8: Epitope Prediction (T/B Cell) ----------
elif page in ["T cell epitope predictor", "B cell epitope predictor"]:
    st.header("Epitope Prediction")
    model_type = "T-cell" if "T" in page else "B-cell"

    organism = st.selectbox("Select Organism", ["Human", "Virus", "Bacteria", "Fungi", "Mouse", "Other"])
    uniprot_id = st.text_input("Enter UniProt ID (optional)")
    sequence = st.text_area("Paste Protein Sequence:", height=200)
    protein_name = st.text_input("Protein Name", "Manual_Protein")

    if st.button("Generate & Predict") and sequence.strip():
        # Generate peptides from protein sequence
        df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)

        # Check the columns in the dataframe
        st.write("Columns in DataFrame:", df.columns)  # Debugging step to check columns
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Load trained model and scaler
        model_file = f"{model_type.lower()}-rf_model.pkl"
        scaler_file = f"{model_type.lower()}-scaler.pkl"

        if not os.path.exists(model_file):
            st.error("Train model first.")
            st.stop()

        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)

        # Ensure all necessary features are present before prediction
        X_pred = scaler.transform(df[FEATURE_COLUMNS])
        df['prediction'] = model.predict(X_pred)

        # Display results
        st.dataframe(df)

        # Plot Immunogenicity Distribution if present
        if 'immunogenicity_score' in df.columns:
            st.subheader("Immunogenicity Distribution")
            st.plotly_chart(px.box(df, y="immunogenicity_score"))
        else:
            st.error("Immunogenicity score column is missing.")

        # Additional plots
        st.subheader("Stability Distribution")
        st.plotly_chart(px.box(df, y="stability"))

        st.subheader("Peptide Length Distribution")
        st.plotly_chart(px.histogram(df, x="peptide_length"))

        st.subheader("Aromaticity Distribution")
        st.plotly_chart(px.violin(df, y="aromaticity", box=True))

        st.subheader("Start/End Positions")
        fig = px.scatter(df, x="start_position", y="end_position", color="prediction")
        st.plotly_chart(fig)

        # Add download button for the CSV
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{protein_name}_predictions.csv")
