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
import os

# Set page config at the very top

st.set_page_config(layout="wide", page_title="Epitope Predictor")

# Add custom CSS background and style
# Inject custom CSS for background and styling

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

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Streamlit Layout and Navigation
st.title("B-cell and T-cell Epitope Predictor")
page = st.sidebar.radio("Navigation", ["Model Training", "T cell epitope predictor", "B cell epitope predictor", "Data Overview"])
df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()


#  Data overview to see the input and output

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

# Train th emodel whenever required to bwt better performance and accuracy

elif page == "Model Training":
    st.header("Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    st.text(f"You selected: {choice}")

    if choice == "B-cell":
        df = df_train_b.copy()
    else:
        df = df_train_t.copy()

    FEATURE_COLUMNS = [
        'protein_seq_length', 'parent_protein_id_length',
        'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
        'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
    ]

    # Drop unwanted columns if they exist
    cols_to_drop = ["parent_protein_id", "protein_seq", "peptide_seq", "start_position", "end_position"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Check for target column
    if "target" not in df.columns:
        st.error("The dataset does not contain the 'target' column required for training.")
        st.stop()

    df = df.dropna(subset=['target'])

    # Get only available feature columns
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    if not available_features:
        st.error("None of the required feature columns are present in the data.")
        st.stop()

    X = df[available_features]
    Y = df["target"]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if st.checkbox("Apply SMOTE for balancing"):
        from imblearn.over_sampling import SMOTE
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

#  Define the function for Epitope prediction and visualization with plot and CSV file

# ... [Keep everything above this line unchanged] ...

elif page == "T cell epitope predictor" or page == "B cell epitope predictor":
    st.header("Epitope Predictor")

    organism = st.selectbox("Select Organism", ["Human", "Bacteria", "Virus", "Fungi", "Mice", "Other"])
    uniprot_id = st.text_input("Enter UniProt ID (Optional)")

    st.text(f"You selected: {organism}")

    # Define default sequence
    default_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVL..."

    # Initialize sequence and protein name
    sequence = ""
    protein_name = "Unknown"

    # Try fetching sequence if UniProt ID is provided
    if uniprot_id:
        sequence, protein_name = fetch_sequence_from_uniprot(uniprot_id)
        if not sequence:
            st.warning("Could not fetch sequence from UniProt. Please paste sequence manually below.")

    # If sequence is still not available, use manual entry
    if not sequence:
        sequence = st.text_area("Paste Protein Sequence:", default_seq, height=200)
        protein_name = st.text_input("Protein Name", "Manual_Protein")

    # Choose prediction type from context
    model_type = "T-cell" if page == "T cell epitope predictor" else "B-cell"

    # Prediction block
    if sequence.strip() != "":
        if st.button("Generate Epitopes and Predict"):
            try:
                df = simulate_peptide_data(sequence, parent_id=protein_name, organism=organism)
                df_features = add_features(df)

                feature_cols = [
                    'protein_seq_length', 'parent_protein_id_length',
                    'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
                    'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
                ]

                model_file = f"{model_type.lower()}-rf_model.pkl"
                scaler_file = f"{model_type.lower()}-scaler.pkl"

                if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                    st.error(f"Model or scaler file not found. Please train the model first.")
                else:
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)

                    X_pred = df_features[feature_cols]
                    X_scaled = scaler.transform(X_pred)
                    predictions = model.predict(X_scaled)

                    df_features['prediction'] = predictions

                    st.success(f"Predicted {len(df_features)} peptides.")
                    st.dataframe(df_features)

                    # --- Visualizations ---
                    st.subheader("Immunogenicity Score Distribution")
                    fig1 = px.violin(df_features, y="immunogenicity_score", box=True, points="all", 
                                     title="Immunogenicity Score", color_discrete_sequence=["#FF6F61"])
                    st.plotly_chart(fig1, use_container_width=True)

                    st.subheader("Hydrophobicity Distribution")
                    fig2 = px.box(df_features, y="hydrophobicity", title="Hydrophobicity", 
                                  color_discrete_sequence=["#66C2A5"])
                    st.plotly_chart(fig2, use_container_width=True)

                    st.subheader("Feature Correlation Heatmap")
                    corr = df_features[feature_cols].corr()
                    fig3, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                    st.pyplot(fig3)

                    st.subheader("Pairwise Feature Relationships")
                    import warnings
                    warnings.filterwarnings("ignore")
                    fig4 = sns.pairplot(df_features[feature_cols + ['prediction']], hue='prediction', palette='husl')
                    st.pyplot(fig4)

                    st.subheader("Download Predictions as CSV")
                    csv = convert_df_to_csv(df_features)
                    st.download_button(
                        label="Download Epitope Predictions",
                        data=csv,
                        file_name=f"{protein_name}_epitope_predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error in prediction or visualization: {str(e)}")

