# This Python (Pandas) code can be used to predict the T-cell and B-cell epitope using UniProt ID or Protein sequence

# Import all required libraries
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
import os  # Added import os to check for file existence
from sklearn.metrics import confusion_matrix

# Step 1: Upload the CSV files (SARS-CoV-2 and IEDB datasets)
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
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# Step 3: Generate peptides from sequence
def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = []
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length]))
    return peptides

# Step 4: Simulate peptide feature data
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

# Step 5: To predict the epitope Fetch sequence or from UniProt_id from Uniprot for your protein of interest

def fetch_sequence_from_uniprot(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.ok:
        lines = response.text.split("\n")
        seq = "".join(lines[1:])
        name = lines[0].split("|")[-1].strip()
        return seq, name
    return None, None

# Load the datasets in the Streamlit

st.set_page_config(layout="wide")
st.title("B-cell and T-cell Epitope Predictor")
page = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "T cell epitope predictor", "B cell epitope predictor"])
df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

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
    st.subheader("Processed Training Data")
    if st.checkbox("Show B-cell Preprocessing"):
        st.dataframe(add_features(df_train_b).head())
    if st.checkbox("Show T-cell Preprocessing"):
        st.dataframe(add_features(df_train_t).head())

# define the function for epitope prediction

elif page == "Model Training":
    st.header("Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t
    df = add_features(df)

    FEATURE_COLUMNS = [
        'protein_seq_length', 'parent_protein_id_length',
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
        st.success("SMOTE applied")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", 0, 100, 42)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    if st.button("Train Random Forest"):
        model = RandomForestClassifier(n_estimators=500, random_state=random_state) 
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        st.success("Model trained successfully!")
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

# Epitope prediction

elif page == "T cell epitope predictor" or page == "B cell epitope predictor":
    st.header("Epitope Prediction with Model")
    
    organism = st.selectbox("Select Organism", ["Human", "Bacteria", "Virus", "Fungi", "Mice", "Other"])
    uniprot_id = st.text_input("Enter UniProt ID (Optional)")
    default_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVL..."
    sequence = None
    protein_name = "Unknown"

    if uniprot_id:
        sequence, protein_name = fetch_sequence_from_uniprot(uniprot_id)

    if not sequence:
        sequence = st.text_area("Paste Protein Sequence:", default_seq, height=200)
        protein_name = st.text_input("Protein Name", "Manual_Protein")
    
    model_type = "T-cell" if page == "T cell epitope predictor" else "B-cell"

    if st.button("Generate Epitopes and Predict"):
        if sequence.strip() != "":  # Ensure sequence is not empty
            df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)
            df_features = add_features(df)

            feature_cols = [
                'protein_seq_length', 'parent_protein_id_length',
                'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
                'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
            ]

            try:
                # Check if model file exists, else display error
                model_file = f"{model_type.lower()}-rf_model.pkl"
                if not os.path.exists(model_file):
                    st.error(f"Model file '{model_file}' not found. Please train the model first.")
                    
                
                model = joblib.load(model_file)
                scaler = joblib.load(f"{model_type.lower()}-scaler.pkl")

                X_pred = df_features[feature_cols]
                X_scaled = scaler.transform(X_pred)
                predictions = model.predict(X_scaled)

                df_features['prediction'] = predictions

                st.success(f"Predicted {len(df_features)} peptides.")
                st.dataframe(df_features)

               # Visualizations for feature analysis
         try:
               #1. Violin Plot for Immunogenicity Score
              fig = px.violin(df_features, y="immunogenicity_score", box=True, points="all", 
                    title="Immunogenicity Score Distribution", color_discrete_sequence=["#FF6F61"])
              fig.update_layout(
              yaxis_title="Immunogenicity Score", xaxis_title="Distribution", font=dict(size=12)
             )
             st.plotly_chart(fig, use_container_width=True)

            # 2. Box Plot for Hydrophobicity
            fig = px.box(df_features, y="hydrophobicity", title="Hydrophobicity Distribution",
                 color_discrete_sequence=["#66C2A5"])
           fig.update_layout(
           yaxis_title="Hydrophobicity", font=dict(size=12)
           )
           st.plotly_chart(fig, use_container_width=True)

        # 3. Correlation Heatmap of Features (instead of pairplot)
         corr = df_features[feature_cols].corr()  # Compute correlations
         plt.figure(figsize=(10, 8))
         sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
         plt.title("Feature Correlation Heatmap", fontsize=16)
         st.pyplot()  # Pass the matplotlib figure to Streamlit

    # 4. Confusion Matrix Plot (using seaborn heatmap)
        cm = confusion_matrix(Y_test, Y_pred)
        cm_fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axis
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)  # Create heatmap
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix", fontsize=16)
        st.pyplot(cm_fig)  # Pass the confusion matrix figure to st.pyplot()

   except Exception as e:
    st.error(f"Error in prediction or visualization: {str(e)}")
