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

st.title("T-cell Epitope Predictor")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Training", "Epitope Prediction"])

# Load data function
@st.cache_data
def load_data():
    bcell_url = "https://drive.google.com/uc?id=1_v_AiVvwpSnuKCNplAimFS8sOlu-hZeQ&export=download"
    covid_url = "https://drive.google.com/uc?id=13JRk-wG8GggBTA-3J1U4R5x3nhrT7KbY&export=download"
    sars_url = "https://drive.google.com/uc?id=1hlza1PsXkHiBqzhpZpVKcLDlLUs4aQtj&export=download"
    Tcell_url = "https://drive.google.com/uc?id=1wYhEDx7pRxiHzD58R2ihfDrSp5Bu68cc&export=download"
    
    df_1 = pd.read_csv(bcell_url)   # bcells csv
    df_2 = pd.read_csv(Tcell_url)   # tcells csv
    df_3 = pd.read_csv(sars_url)    # sars csv
    df_test = pd.read_csv(covid_url) # covid csv
    
    df_train = pd.concat([df_1, df_3])
    
    return df_1, df_2, df_3, df_test, df_train

# Function to generate peptides
def generate_peptides(sequence, length=9):
    return [(i + 1, i + length, sequence[i:i + length]) for i in range(len(sequence) - length + 1)]

# Function to simulate peptide data
def simulate_peptide_data(seq, parent_id="Spike_SARS_CoV_2"):
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

# Load data
with st.spinner("Loading data..."):
    df_1, df_2, df_3, df_test, df_train = load_data()

if page == "Data Overview":
    st.header("Data Overview")
    
    st.subheader("B-cells Dataset")
    st.dataframe(df_1.head())
    
    st.subheader("T-cells Dataset")
    st.dataframe(df_2.head())
    
    st.subheader("SARS Dataset")
    st.dataframe(df_3.head())
    
    st.subheader("COVID Dataset")
    st.dataframe(df_test.head())
    
    st.subheader("Training Dataset")
    st.dataframe(df_train.head())
    
    # Data preprocessing
    if st.checkbox("Show Data Preprocessing"):
        st.write("Adding feature columns...")
        df_train['protein_seq_length'] = df_train['protein_seq'].astype(str).map(len)
        df_train['peptide_seq_length'] = df_train['peptide_seq'].astype(str).map(len)
        df_train['parent_protein_id_length'] = df_train['parent_protein_id'].astype(str).map(len)
        df_train['peptide_length'] = df_train['end_position'] - df_train['start_position'] + 1
        
        st.dataframe(df_train.head())
    
    # Data visualization
    if st.checkbox("Show Data Visualization"):
        st.write("Visualizing numerical variables...")
        num_vars = [x for x in df_train.columns if df_train[x].dtypes != 'O']
        
        for i in num_vars:
            fig = px.box(df_train, y=i, color='target')
            st.plotly_chart(fig)

elif page == "Model Training":
    st.header("Model Training")
    
    # Prepare data for training
    st.write("Preparing data for training...")
    df_train['protein_seq_length'] = df_train['protein_seq'].astype(str).map(len)
    df_train['peptide_seq_length'] = df_train['peptide_seq'].astype(str).map(len)
    df_train['parent_protein_id_length'] = df_train['parent_protein_id'].astype(str).map(len)
    df_train['peptide_length'] = df_train['end_position'] - df_train['start_position'] + 1
    
    df_train = df_train.drop(["parent_protein_id", "protein_seq", "peptide_seq", "start_position", "end_position"], axis=1)
    df_train = df_train.dropna(subset=['target'])
    
    X = df_train.drop("target", axis=1)
    Y = df_train["target"]
    
    # SMOTE for data upscaling
    if st.checkbox("Apply SMOTE for data upscaling"):
        smote = SMOTE()
        X, Y = smote.fit_resample(X, Y)
        st.success("SMOTE applied successfully!")
    
    # Data normalization
    if st.checkbox("Apply Data Normalization"):
        X = MinMaxScaler().fit_transform(X)
        st.success("Data normalized successfully!")
    
    # Train-test split
    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random state", 0, 100, 42)
    
    if st.button("Split Data"):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        st.write(f'Training Features Shape: {X_train.shape}')
        st.write(f'Testing Features Shape: {X_test.shape}')
        st.write(f'Training Labels Shape: {Y_train.shape}')
        st.write(f'Testing Labels Shape: {Y_test.shape}')
    
    # Train model
    n_estimators = st.slider("Number of estimators", 100, 2000, 1000)
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            rf.fit(X_train, Y_train)
            Y_pred_rf = rf.predict(X_test)
            
            st.success("Model trained successfully!")
            st.write("Accuracy:", accuracy_score(Y_test, Y_pred_rf))
            st.text("Classification Report:")
            st.text(classification_report(Y_test, Y_pred_rf))
            
            # Confusion Matrix
            cm = confusion_matrix(Y_test, Y_pred_rf)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

elif page == "Epitope Prediction":
    st.header("T-cell Epitope Predictor")
    
    # Default spike protein sequence
    default_seq = ("MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHV"
                  "SGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFL"
                  "GVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINL"
                  "VRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENG"
                  "TITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWN"
                  "RKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKL"
                  "PDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYG"
                  "FQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGR"
                  "DIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPRWVYSTG"
                  "SNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNN"
                  "SIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEV"
                  "
