# This Python (Pandas) code can be used to predict the Tcell and B cell epitoe using Uni_prot ID or Protein sequence
# # Import all required libraries

# importing all the required library
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
from Bio import SeqIO
import random
import joblib
import requests


# upload all the data data after cleaning
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

# Define all the parameter or features we want to see
def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['peptide_seq_length'] = df['peptide_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# Generate the peptide sequence by simulating the data.  Define the function.
def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

def simulate_peptide_data(seq, parent_id="Unknown_Protein"):
    peptides = generate_peptides(seq)
    rows = []
    for start, end, pep in peptides:
        analysis = ProteinAnalysis(pep)
        try:
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
        except:
            continue
    return pd.DataFrame(rows)

# Extract protein sequence from UniProt ID
def fetch_sequence_from_uniprot(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.ok:
        fasta = response.text
        lines = fasta.splitlines()
        sequence = "".join(lines[1:])
        name = lines[0].split("|")[2].split()[0]
        return sequence, name
    else:
        return None, None

# Run the app Streamlit
st.set_page_config(layout="wide")
st.title("🔬 B-cell and T-cell Epitope Predictor")

page = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Epitope Prediction"])

df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

# Data overview in the Streamlit app.
if page == "Data Overview":
    st.header("📊 Data Overview")
    st.subheader("B-cell Dataset")
    st.dataframe(df_bcell.head())
    st.subheader("T-cell Dataset")
    st.dataframe(df_tcell.head())
    st.subheader("SARS Dataset")
    st.dataframe(df_sars.head())
    st.subheader("COVID Test Dataset")
    st.dataframe(df_test.head())
    st.subheader("Processed Training Data")
    if st.checkbox("Show B-cell Preprocessing"):
        st.dataframe(add_features(df_train_b).head())
    if st.checkbox("Show T-cell Preprocessing"):
        st.dataframe(add_features(df_train_t).head())

# Train the model before prediction
elif page == "Model Training":
    st.header("🤖 Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t
    df = add_features(df)

    FEATURE_COLUMNS = [
        'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
        'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
        'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
    ]

    df = df.drop(["parent_protein_id", "protein_seq", "peptide_seq", "start_position", "end_position"], axis=1)
    df = df.dropna(subset=['target'])

    X = df[FEATURE_COLUMNS]
    Y = df["target"]

    if st.checkbox("Apply SMOTE for balancing"):
        smote = SMOTE()
        X, Y = smote.fit_resample(X, Y)
        st.success("✅ SMOTE applied")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    st.success("✅ Features normalized")

    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", 0, 100, 42)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    if st.button("Train Random Forest"):
        model = RandomForestClassifier(n_estimators=500, random_state=random_state)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        st.success("🎉 Model trained successfully!")
        st.write("Accuracy:", accuracy_score(Y_test, Y_pred))
        st.text("Classification Report:")
        st.text(classification_report(Y_test, Y_pred))

        cm = confusion_matrix(Y_test, Y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{choice.lower()}-scaler.pkl")
        st.success(f"Model and Scaler saved as '{choice.lower()}-rf_model.pkl' and '{choice.lower()}-scaler.pkl'")

# Add all the features and condition for Epitope prediction
if page == "Epitope Prediction":
    st.header("🔎 Epitope Prediction with Model")

    prediction_type = st.radio("Epitope Type", ["B-cell", "T-cell"])
    organism = st.selectbox("Select Organism", ["Human", "Virus", "Bacteria", "Fungus", "Other"])

    input_method = st.radio("Input Method", ["UniProt ID", "Paste Protein Sequence"])
    if input_method == "UniProt ID":
        uniprot_id = st.text_input("Enter UniProt ID (e.g., P0DTC2)")
        if st.button("Fetch Sequence"):
            sequence, name = fetch_sequence_from_uniprot(uniprot_id)
            if sequence:
                st.success(f"Protein Name: {name}")
                st.text_area("Protein Sequence", value=sequence, height=150)
            else:
                st.error("Invalid UniProt ID or data not found.")
    else:
        sequence = st.text_area("Paste Protein Sequence", height=150)
        name = st.text_input("Enter Protein Name", "Unknown_Protein")

    if st.button("Generate Epitopes and Predict"):
        if sequence:
            df = simulate_peptide_data(sequence, parent_id=name)
            df_features = add_features(df)
            df_features['organism'] = organism

            feature_cols = [
                'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length', 'peptide_length',
                'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker',
                'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
            ]

            try:
                model = joblib.load(f"{prediction_type.lower()}-rf_model.pkl")
                scaler = joblib.load(f"{prediction_type.lower()}-scaler.pkl")
                X_pred = scaler.transform(df_features[feature_cols])
                df_features['immunogenicity'] = model.predict(X_pred)

                st.success(f"✅ Predicted {len(df_features)} peptides.")
                st.dataframe(df_features)

                st.subheader("📊 Feature Distributions")
                for feature in feature_cols:
                    st.plotly_chart(px.histogram(df_features, x=feature, color="immunogenicity", title=feature))

                csv = df_features.to_csv(index=False)
                st.download_button("Download CSV", data=csv, file_name="predicted_epitopes.csv")

            except Exception as e:
                st.error(f"Model or Scaler not found. Error: {e}")
