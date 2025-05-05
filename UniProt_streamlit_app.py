# Import all the required library for this project
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import Entrez
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
import requests
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Step 1:Fetch Protein Sequence or accession_id from UniProt API for your gene of interest.

def get_uniprot_sequence(protein_id):
    """
    Fetch the protein sequence from UniProt based on the given UniProt ID.
    """
    base_url = f'https://www.uniprot.org/uniprot/{protein_id}.fasta'
    response = requests.get(base_url)

    if response.status_code == 200:
        fasta_data = response.text
        sequence = "".join(fasta_data.split("\n")[1:])  # Remove header line
        return sequence
    else:
        st.error("Failed to fetch the sequence from UniProt. Please check the protein ID.")
        return None

# Step 2: Create or define the python function to generate the peptide sequence 

def generate_peptides(sequence, min_length=9, max_length=14):
    """
    Generate peptides from the given sequence by sliding window approach.
    """
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

# Step 3: After defining the function based on the features, Simulate Peptide Data

def simulate_peptide_data(seq, parent_id="Unknown"):
    """
    Simulate peptide data by analyzing the sequence and computing different features.
    """
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

# Step 4: Model Training and Prediction of the epitope

def add_features(df):
    """
    Add feature columns like sequence lengths and various computed properties.
    """
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['peptide_seq_length'] = df['peptide_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# Step 5: Run the Model training in the app (if necessary)

def train_model(df):
    """
    Train a Random Forest model using features derived from the peptide data.
    """
    df = add_features(df)
    FEATURE_COLUMNS = [
        'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
        'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
        'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
    ]
    
    X = df[FEATURE_COLUMNS]
    Y = df["target"]  # Assuming there's a 'target' column for classification

    # Feature scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    # Train the model
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, Y_train)

    # Evaluate of the model after training
    
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(Y_test, Y_pred))

    # Once you train and evaluate save the model for future use
    
    joblib.dump(model, 'epitope_prediction_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# Step 6: Streamlit Interface for User Input
st.set_page_config(layout="wide")
st.title("🔬 Epitope Prediction and Visualization Tool")

page = st.sidebar.radio("Navigation", ["Predict Epitope", "Model Training"])

# Creat the Epitope prediction page

if page == "Predict Epitope":
    st.header("🔍 Predict Epitope from UniProt ID")

    protein_id = st.text_input("Enter UniProt Protein ID (e.g., P0DTC2):")

    if protein_id:
        sequence = get_uniprot_sequence(protein_id)
        
        if sequence:
            st.write(f"Protein Sequence for {protein_id}:")
            st.text(sequence)

            df = simulate_peptide_data(sequence, parent_id=protein_id)
            df_features = add_features(df)

            FEATURE_COLUMNS = [
                'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
                'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
                'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
            ]
            X_pred = df_features[FEATURE_COLUMNS]

            # Load trained model
            
            model = joblib.load('epitope_prediction_model.pkl')
            scaler = joblib.load('scaler.pkl')

            # Predict
            
            X_pred_scaled = scaler.transform(X_pred)
            predictions = model.predict(X_pred_scaled)

            df['Prediction'] = predictions
            st.write(df)

            # Visualize Predictions
            
            fig = plt.figure(figsize=(10, 6))
            sns.countplot(x='Prediction', data=df)
            st.pyplot(fig)

            st.write("🔬 Peptides and Predicted Epitopes have been displayed.")

# Model Training Page

elif page == "Model Training":
    st.header("🤖 Train Epitope Prediction Model")

    # Load and show datasets
    if st.button("Train Model"):
        st.write("Training model with the B-cell/T-cell data...")
        # Assume you have a pre-existing dataset (example 'df_train_b')
        # train_model(df_train_b)  # Replace with your training data
