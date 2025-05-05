
# Step 1: Importing Required Libraries
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

# Step 2: Helper Functions

# Function to fetch protein sequence from UniProt based on ID
def get_uniprot_sequence(protein_id):
    url = f'https://www.uniprot.org/uniprot/{protein_id}.fasta'
    response = requests.get(url)
    if response.status_code == 200:
        return "".join(response.text.splitlines()[1:])
    else:
        st.error(f"Could not retrieve sequence for {protein_id}. Please check the UniProt ID.")
        return None

# Function to add features to the dataframe
def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['peptide_seq_length'] = df['peptide_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# Function to simulate peptide data from a protein sequence
def generate_peptides(sequence, min_length=9, max_length=14):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

def simulate_peptide_data(seq, parent_id="Unknown"):
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

# Step 3: Load Data (Simulated or from URL) and Model Training

def load_data():
    # Load any external or pre-defined dataset
    pass

def train_model(df, target_column):
    # Define the feature columns for prediction
    FEATURE_COLUMNS = [
        'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
        'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
        'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
    ]

    df = add_features(df)
    X = df[FEATURE_COLUMNS]
    Y = df[target_column]

    # Split data for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # Apply SMOTE for class balancing if needed
    smote = SMOTE()
    X_train, Y_train = smote.fit_resample(X_train, Y_train)

    # Model Training (Random Forest)
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, Y_train)

    # Save the trained model and scaler
    joblib.dump(model, 'epitope_prediction_model.pkl')
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'scaler.pkl')

    return model, scaler

# Step 4: Epitope Prediction and Results Visualization

# Function to predict epitopes based on user input
def predict_epitope(model, scaler, sequence, feature_columns):
    df = simulate_peptide_data(sequence)
    df_features = add_features(df)
    X_pred = df_features[feature_columns]
    X_pred_scaled = scaler.transform(X_pred)
    predictions = model.predict(X_pred_scaled)
    df_features['prediction'] = predictions
    return df_features

# Step 5: Streamlit Application

# Setup Streamlit App
st.set_page_config(layout="wide")
st.title("🔬 Epitope Predictor")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Epitope Prediction"])

if page == "Data Overview":
    st.header("📊 Data Overview")
    st.subheader("B-cell and T-cell Dataset")

    # Add your dataset or mock data preview here
    # st.dataframe(df)  # Show sample data

elif page == "Model Training":
    st.header("🤖 Model Training")
    choice = st.selectbox("Select Epitope Type", ["B-cell", "T-cell"])

    if st.button("Train Model"):
        # Replace 'df_train_b' and 'df_train_t' with your actual dataset
        model, scaler = train_model(df_train_b if choice == "B-cell" else df_train_t, 'target')  # Target should be the column for prediction
        st.success(f"Model trained and saved as {choice.lower()}-epitope_prediction_model.pkl")

elif page == "Epitope Prediction":
    st.header("🔍 Epitope Prediction")

    epitope_type = st.selectbox("Select Epitope Type", ["B-cell", "T-cell"])
    input_type = st.selectbox("Input Type", ["UniProt ID", "Protein Sequence"])

    if input_type == "UniProt ID":
        protein_id = st.text_input("Enter UniProt Protein ID (e.g., P0DTC2):")
        if protein_id:
            sequence = get_uniprot_sequence(protein_id)
    else:
        sequence = st.text_area("Enter Protein Sequence (Amino acids)", height=200)

    if st.button("Generate Epitopes and Predict"):
        if sequence:
            # Load the model and scaler
            model = joblib.load(f'{epitope_type.lower()}-epitope_prediction_model.pkl')
            scaler = joblib.load(f'{epitope_type.lower()}-scaler.pkl')

            # Predict the epitopes
            feature_columns = [
                'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
                'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
                'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
            ]
            df_results = predict_epitope(model, scaler, sequence, feature_columns)

            # Show the predicted results
            st.write("Predicted Epitopes")
            st.dataframe(df_results)

            # Plot feature distributions
            st.subheader("Feature Distributions")
            st.plotly_chart(px.histogram(df_results, x="peptide_length", title="Peptide Length Distribution"))
            st.plotly_chart(px.histogram(df_results, x="hydrophobicity", title="Hydrophobicity Distribution"))

            # Show the predicted epitope class distribution
            st.subheader("Prediction Class Distribution")
            fig = plt.figure(figsize=(10, 6))
            sns.countplot(x='prediction', data=df_results)
            st.pyplot(fig)

            # Provide CSV download button
            csv = df_results.to_csv(index=False)
            st.download_button("Download Results as CSV", data=csv, file_name="predicted_epitopes.csv")

