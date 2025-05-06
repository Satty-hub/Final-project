# This Python (Pandas) code can be used to predict the Tcell and B cell epitoe using Uni_prot ID or Protein sequence
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

# Step 1: Upload the csv file from the download. I have used sars covid data for training. I used IEDB and UniProt to extract the B cell and T cell epitope seq for training the model.
@st.cache_data
def load_data():
    bcell_url = "https://drive.google.com/uc?id=1_v_AiVvwpSnuKCNplAimFS8sOlu-hZeQ&export=download"
    covid_url = "https://drive.google.com/uc?id=13JRk-wG8GggBTA-3J1U4R5x3nhrT7KbY&export=download"
    sars_url = "https://drive.google.com/uc?id=1hlza1PsXkHiBqzhpZpVKcLDlLUs4aQtj&export=download"
    Tcell_url = "https://drive.google.com/uc?id=1wYhEDx7pRxiHzD58R2ihfDrSp5Bu68cc&export=download"

    df_bcell = pd.read_csv(bcell_url) # sars b cell data
    df_tcell = pd.read_csv(Tcell_url) # sars b cell data
    df_sars = pd.read_csv(sars_url) # sars data
    df_test = pd.read_csv(covid_url) # SARS-CoV-2 cell data

    df_train_b = pd.concat([df_bcell, df_sars])
    df_train_t = pd.concat([df_tcell, df_sars])

    return df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t  

# Step 2: define all the feature what we want to use to get the data for future 
def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# Step 3: Define the dataframe for peptides generation

def generate_peptides(sequence, min_length=8, max_length=11):
    peptides = [] # empty list where predicted peptides list should store back
    for length in range(min_length, max_length + 1):
        for i in range(len(sequence) - length + 1):
            peptides.append((i + 1, i + length, sequence[i:i + length])) # generate all the peptide and append into the list 
    return peptides

# Step 4: Simulate input data to add all the feature and condition we want to include in the peptides list

def simulate_peptide_data(seq, parent_id="Unknown", organism="Unknown"):  # parent_id and and organism is unknown and it will be decided based on the input.
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY") # sometime amino acids sequence are not with right code or na so it is better to valid the sequence to avaoid the error while generating the peptide seq
    peptides = generate_peptides(seq)
    rows = []

    for start, end, pep in peptides:
        # avaoid the peptides seq with invalid amino acids
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
            # Skip peptides that still cause errors
            continue

    return pd.DataFrame(rows)

# Step 5: Define the function to directly Fetch the UniProt_id from UniProt or protein_seq we provide

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

# Step 6: Once data is fetch for input then we can run the Streamlit 

st.set_page_config(layout="wide") # font and side bar menu
st.title("B-cell and T-cell Epitope Predictor") # this is the title

page = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "T cell epitope predictor", "B cell epitope predictor"]) # These are the navigator
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

#  Train the model for prediction

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
    
    df = df.drop(["parent_protein_id", "protein_seq", "peptide_seq", "start_position", "end_position"], axis=1) # If there is any duplicate seq
    df = df.dropna(subset=['target'])

    X = df[FEATURE_COLUMNS]
    Y = df["target"]

    if st.checkbox("Apply SMOTE for balancing"): # To avaoid the incorrect value to ake it more balancing and also to avoid more outlier
        smote = SMOTE()
        X, Y = smote.fit_resample(X, Y)
        st.success(" SMOTE applied")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", 0, 100, 42)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state) #  Train the model before so we can get better prediction based on the input 

    if st.button("Train Random Forest"):
        model = RandomForestClassifier(n_estimators=500, random_state=random_state) 
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        st.success(" Model trained successfully!")
        st.write("Accuracy:", accuracy_score(Y_test, Y_pred)) # Provide the training report
        st.text("Classification Report:")
        st.text(classification_report(Y_test, Y_pred))

        cm = confusion_matrix(Y_test, Y_pred)    # Provide the training report heatmap
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{choice.lower()}-scaler.pkl")
        st.success(f"Model and Scaler saved as '{choice.lower()}-rf_model.pkl' and '{choice.lower()}-scaler.pkl'")


# Input for the Prediction model

elif page in ["T cell epitope predictor", "B cell epitope predictor"]:
    st.header("Epitope Prediction")
    
    organism = st.selectbox("Select Organism", ["Human", "Bacteria", "Virus", "Fungi", "Mice", "Other"]) # Input you want to choose from
    uniprot_id = st.text_input("Enter UniProt ID (Optional)")
    default_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVL..."
    sequence = None
    protein_name = "Unknown"

    if uniprot_id:
        sequence, protein_name = fetch_sequence_from_uniprot(uniprot_id)

    if not sequence:
        sequence = st.text_area("Paste Protein Sequence:", default_seq, height=200)
        protein_name = st.text_input("Protein Name", "Manual_Protein")
    
    # Set model type based on navigation choice
    model_type = "T-cell" if page == "T cell epitope predictor" else "B-cell"

    if st.button("Generate Epitopes and Predict"):  # Button to trigger the prediction
        if sequence.strip() != "":  # Ensure sequence is not empty
            df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)
            df_features = add_features(df)

            feature_cols = [
                'protein_seq_length', 'parent_protein_id_length',
                'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
                'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
            ]

            try:
                # Load the pre-trained model and scaler based on the selected model type
                model = joblib.load(f"{model_type.lower()}-rf_model.pkl")
                scaler = joblib.load(f"{model_type.lower()}-scaler.pkl")

                X_pred = df_features[feature_cols]
                X_scaled = scaler.transform(X_pred)
                predictions = model.predict(X_scaled)

                df_features['prediction'] = predictions

                # Display success message and the predictions table
                
                st.success(f"Predicted {len(df_features)} peptides.")
                st.dataframe(df_features)

                # Feature distributions visualization
                
                st.subheader("Peptide Feature Distributions")
                feature_cols_to_plot = [
                    'peptide_length', 'hydrophobicity', 'isoelectric_point', 'stability',
                    'aromaticity', 'emini', 'kolaskar_tongaonkar', 'chou_fasman',
                    'parker', 'immunogenicity_score'
                ]

                for col in feature_cols_to_plot:
                    if col in df_features.columns:
                        fig = px.histogram(df_features, x=col, nbins=20, title=f'Distribution of {col}')
                        st.plotly_chart(fig, use_container_width=True)

                # Positive predictions stats
                
                positive_preds = df_features[df_features['prediction'] == 1]
                st.subheader(f"{model_type} Epitope Summary")
                st.metric("Number of Predicted Epitopes", len(positive_preds))
                st.metric("Average Epitope Length", f"{positive_preds['peptide_length'].mean():.2f}")
                st.metric("Total Epitope Length", f"{positive_preds['peptide_length'].sum():.2f}")

                # Epitope length histogram
                
                st.plotly_chart(px.histogram(positive_preds, x='peptide_length', nbins=10,
                                             title=f'{model_type} Epitope Length Distribution'))

                # Allow file download for the predicted epitopes
                
                csv = df_features.to_csv(index=False)
                st.download_button("Download Predicted CSV", data=csv, file_name="predicted_epitopes.csv")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
