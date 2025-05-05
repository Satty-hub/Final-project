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
import joblib
import random

# Consistent feature list
feature_cols = [
    'protein_seq_length', 'peptide_seq_length', 'parent_protein_id_length',
    'peptide_length', 'chou_fasman', 'emini', 'kolaskar_tongaonkar',
    'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'
]

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

def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['peptide_seq_length'] = df['peptide_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

def generate_peptides(sequence, length=9):
    return [(i + 1, i + length, sequence[i:i + length]) for i in range(len(sequence) - length + 1)]

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

# Streamlit app
st.set_page_config(layout="wide")
st.title("🔬 B-cell and T-cell Epitope Predictor")
page = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Epitope Prediction"])

df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

if page == "Data Overview":
    st.header("📊 Data Overview")
    st.dataframe(df_bcell.head())
    st.dataframe(df_tcell.head())

elif page == "Model Training":
    st.header("🤖 Model Training")
    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t
    df = add_features(df)
    df = df.drop(["parent_protein_id", "protein_seq", "peptide_seq", "start_position", "end_position"], axis=1)
    df = df.dropna(subset=['target'])

    X = df[FEATURE_COLUMNS]
    Y = df["target"]

    if st.checkbox("Apply SMOTE"):
        X, Y = SMOTE().fit_resample(X, Y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", 0, 100, 42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    if st.button("Train Random Forest"):
        model = RandomForestClassifier(n_estimators=500, random_state=random_state)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        st.success("🎉 Model trained successfully!")
        st.write("Accuracy:", accuracy_score(Y_test, Y_pred))
        st.text(classification_report(Y_test, Y_pred))

        cm = confusion_matrix(Y_test, Y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        joblib.dump(model, f"{choice.lower()}-rf_model.pkl")
        joblib.dump(scaler, f"{choice.lower()}-scaler.pkl")
        st.success("✅ Model and scaler saved.")

elif page == "Epitope Prediction":
    st.header("🔎 Epitope Prediction")
    default_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHV"
    sequence = st.text_area("Paste Protein Sequence:", default_seq, height=150)

    if st.button("Generate Epitopes and Predict"):
        if sequence:
            df = simulate_peptide_data(sequence)
            df_features = add_features(df)

            X_pred = df_features[feature_cols]
            X_scaled = scaler.transform(X_pred)
            df_features['prediction'] = model.predict(X_scaled)

            try:
                model = joblib.load("b-cell-rf_model.pkl")
                scaler = joblib.load("b-cell-scaler.pkl")
                X_scaled = scaler.transform(X_pred)
                df['prediction'] = model.predict(X_scaled)

                st.success(f"Predicted {len(df_features)} peptides")
                st.dataframe(df_features)

                st.subheader("📈 Peptide Property Distributions")
                st.plotly_chart(px.histogram(df_features, x="peptide_length", title="Peptide Length"))
                st.plotly_chart(px.histogram(df_features, x="hydrophobicity", title="Hydrophobicity"))
                st.plotly_chart(px.histogram(df_features, x="isoelectric_point", title="Isoelectric Point"))

                csv = df_features.to_csv(index=False)
                st.download_button("Download CSV", data=csv, file_name="predicted_epitopes.csv")

            except Exception as e:
                st.error(f"❗ Model and Scaler files missing or error: {e}")
        else:
            st.error("❗ Please enter a valid sequence.")

