# This Python (Pandas) code can be used to predict the Tcell and B cell epitoe using Uni_prot ID or Protein sequence
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
import random
import joblib
import requests


# upload all the data data after cleaning
# I have used here sars data set for training purposes. I have downloaded the sequence from IEDB for train the model and define all the function.

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


# Function to fetch protein sequence from UniProt based on ID

def get_uniprot_sequence(protein_id):
    url = f'https://www.uniprot.org/uniprot/{protein_id}.fasta'
    response = requests.get(url)
    if response.status_code == 200:
        return "".join(response.text.splitlines()[1:])
    else:
        st.error(f"Could not retrieve sequence for {protein_id}. Please check the UniProt ID.")
        return None


# Define all the parameter or features we want to see

def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['peptide_seq_length'] = df['peptide_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df


# Generate the peptide sequence by simulating the data. Define the function.

def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

# Simulation fo the peptide

def simulate_peptide_data(seq, parent_id="Spike_SARS_CoV_2"):
    # provide the amino acids list with valid code
    
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    # Filter the sequence to only contain valid amino acids code
    
    seq = ''.join([aa for aa in seq if aa in valid_aa])

    if len(seq) == 0:
        raise ValueError("The sequence contains no valid amino acids.")

    peptides = generate_peptides(seq, min_length=8, max_length=11)  # Peptides between 8 and 11 amino acids
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

    # Define the feature columns
    
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

        # Save model and scaler
        
        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{choice.lower()}-scaler.pkl")
        st.success(f"Model and Scaler saved as '{choice.lower()}-rf_model.pkl' and '{choice.lower()}-scaler.pkl'")

# Add all the features and condition for Epitope prediction

if page == "Epitope Prediction":
    st.header("🔎 Epitope Prediction with Model")

    # Input option: UniProt ID or Protein Sequence
    input_type = st.selectbox("Input Type", ["UniProt ID", "Protein Sequence"])

    if input_type == "UniProt ID":
        protein_id = st.text_input("Enter UniProt Protein ID (e.g., P0DTC2):")
        if protein_id:
            sequence = get_uniprot_sequence(protein_id)
    else:
        sequence = st.text_area("Paste Protein Sequence:", height=200)

    if st.button("Generate Epitopes and Predict"):
        if sequence:
            with st.spinner("Generating peptides and predicting..."):
                df = simulate_peptide_data(sequence)
                df_features = add_features(df)

                # Define the feature columns
                feature_cols = [
                    'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
                    'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
                    'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
                ]

                try:
                    model = joblib.load("b-cell-rf_model.pkl")
                    scaler = joblib.load("b-cell-scaler.pkl")
                    
                    X_pred = df_features[feature_cols]
                    X_scaled = scaler.transform(X_pred)
                    predictions = model.predict(X_scaled)

                    df_features['prediction'] = predictions

                    st.success(f"✅ Predicted {len(df_features)} peptides.")
                    st.dataframe(df_features)

                    # Display feature distribution graphs
                    st.subheader("📈 Feature Distributions")

                    # Plot histograms for all features
                    for feature in feature_cols:
                        st.plotly_chart(px.histogram(df_features, x=feature, title=f"Distribution of {feature}"))

                    # Plot distribution for 'prediction' if it's available
                    if 'prediction' in df_features.columns:
                        st.plotly_chart(px.histogram(df_features, x="prediction", title="Prediction Distribution"))

                    # Option to download the predictions as a CSV file
                    csv = df_features.to_csv(index=False)
                    st.download_button("Download CSV", data=csv, file_name="predicted_epitopes.csv")

                except Exception as e:
                    st.error(f"❗ Model and Scaler files missing or error: {e}")
