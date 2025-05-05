
# ------------------------------------------
# STEP 0: IMPORT LIBRARIES
# ------------------------------------------
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

# ------------------------------------------
# STEP 1: LOAD AND CACHE DATA
# ------------------------------------------
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

# ------------------------------------------
# STEP 2: ADD FEATURES TO DATA
# ------------------------------------------
def add_features(df):
    df = df.copy()
    df['protein_seq_length'] = df['protein_seq'].astype(str).map(len)
    df['peptide_seq_length'] = df['peptide_seq'].astype(str).map(len)
    df['parent_protein_id_length'] = df['parent_protein_id'].astype(str).map(len)
    df['peptide_length'] = df['end_position'] - df['start_position'] + 1
    return df

# ------------------------------------------
# STEP 3: EPITOPE GENERATION (SIMULATED)
# ------------------------------------------
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

# ------------------------------------------
# STEP 4: APP LAYOUT
# ------------------------------------------
st.set_page_config(layout="wide")
st.title("🔬 B-cell and T-cell Epitope Predictor")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Epitope Prediction"])

# Load data
with st.spinner("Loading data..."):
    df_bcell, df_tcell, df_sars, df_test, df_train_b, df_train_t = load_data()

# ------------------------------------------
# PAGE 1: DATA OVERVIEW
# ------------------------------------------
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
        df_bcell_processed = add_features(df_train_b)
        st.dataframe(df_bcell_processed.head())

    if st.checkbox("Show T-cell Preprocessing"):
        df_tcell_processed = add_features(df_train_t)
        st.dataframe(df_tcell_processed.head())

# ------------------------------------------
# PAGE 2: MODEL TRAINING
# ------------------------------------------
elif page == "Model Training":
    st.header("🤖 Model Training")

    choice = st.selectbox("Select Prediction Type", ["B-cell", "T-cell"])
    df = df_train_b if choice == "B-cell" else df_train_t
    df = add_features(df)

    # Drop unused columns
    df = df.drop(["parent_protein_id", "protein_seq", "peptide_seq", "start_position", "end_position"], axis=1)
    df = df.dropna(subset=['target'])

    # Split data
    X = df.drop("target", axis=1)
    Y = df["target"]

    if st.checkbox("Apply SMOTE for balancing"):
        smote = SMOTE()
        X, Y = smote.fit_resample(X, Y)
        st.success("✅ SMOTE applied")

    if st.checkbox("Normalize features"):
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        st.success("✅ Normalization applied")

    test_size = st.slider("Test size", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", 0, 100, 42)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    if st.button("Train Random Forest"):
        with st.spinner("Training..."):
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

# ------------------------------------------
# PAGE 3: EPITOPE PREDICTION
# ------------------------------------------

elif page == "Epitope Prediction":
    st.header("🔎 Epitope Prediction (Actual Model-Based)")

    @st.cache_resource
    def load_model_scaler():
        try:
            model = joblib.load("epitope_model.pkl")
            scaler = joblib.load("epitope_scaler.pkl")
            return model, scaler
        except Exception as e:
            st.error(f"⚠️ Failed to load model/scaler: {e}")
            return None, None

    model, scaler = load_model_scaler()

    default_seq = (
        "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHV"
        "SGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFL"
        "GVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINL"
    )
    sequence = st.text_area("Paste Protein Sequence:", default_seq, height=200)

    if st.button("Predict Epitopes"):
        if not sequence:
            st.error("❗ Please enter a sequence.")
        elif model is None or scaler is None:
            st.error("❗ Model and Scaler files are missing.")
        else:
            with st.spinner("🔬 Extracting features and predicting..."):
                df = simulate_peptide_data(sequence)
                df["peptide_length"] = df["end_position"] - df["start_position"] + 1

                # Feature preparation
                features = [
                    "chou_fasman", "emini", "kolaskar_tongaonkar", "parker",
                    "isoelectric_point", "aromaticity", "hydrophobicity", "stability"
                ]
                X = df[features]
                X_scaled = scaler.transform(X)

                # Prediction
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:, 1]

                df["Prediction"] = ["Epitope" if p == 1 else "Non-Epitope" for p in preds]
                df["Probability"] = np.round(probs, 3)

                st.success(f"✅ Predicted {len(df)} peptides")
                st.dataframe(df)

                # --- PLOTS ---
                st.subheader("📈 Prediction Summary and Visualizations")

                col1, col2 = st.columns(2)

                with col1:
                    fig1 = px.histogram(df, x="peptide_length", color="Prediction",
                                        title="Peptide Length Distribution")
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    fig2 = px.pie(df, names="Prediction", title="Prediction Distribution")
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### 🔬 Biochemical Properties Comparison")

                fig3 = px.scatter(df, x="hydrophobicity", y="stability",
                                  color="Prediction", size="Probability",
                                  hover_data=["peptide_seq"],
                                  title="Hydrophobicity vs Stability")
                st.plotly_chart(fig3, use_container_width=True)

                box_features = ["isoelectric_point", "aromaticity", "hydrophobicity", "stability"]
                for feature in box_features:
                    fig_box = px.box(df, x="Prediction", y=feature,
                                     title=f"{feature} by Prediction")
                    st.plotly_chart(fig_box, use_container_width=True)

                # Download option
                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download Predictions CSV", data=csv, file_name="epitope_predictions.csv", mime="text/csv")
