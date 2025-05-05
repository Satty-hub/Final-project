 B-cell and T-cell Prediction tools for vaccine development
 This is an simple epitope or peptide prediction tool built on Streamlit, which can predicts B-cell and T-cell epitopes from any given protein sequences. It is very easy to use simple by uploading a sequence or protein of interest or using a UniProt ID to extract the sequence directly and uses these sequences to predict the epitopes by applying a trained machine learning model.
 Basic features of the tool: 
 B-cell and T-cell Epitope Prediction: Users can choose between B-cell and T-cell epitope prediction.

Protein Sequence or UniProt ID Input: Users can either paste a protein sequence or enter a UniProt ID, and the sequence is extracted directly from UniProt.

Data Overview: Explore and visualize B-cell and T-cell datasets.

Model Training: Train a Random Forest model to predict epitopes based on features like peptide length, hydrophobicity, isoelectric point, etc.

Peptide Generation: Generate peptides of lengths between 8 to 11 amino acids from the provided protein sequence.

Feature Visualization: Visualize the distributions of features like peptide length, hydrophobicity, stability, etc.

Download Predictions: After predicting epitopes, download the results as a CSV file.

Technologies Used
Streamlit: For creating the web-based interface.

scikit-learn: For training machine learning models.

BioPython: For sequence analysis (e.g., calculating isoelectric point, hydrophobicity).

SMOTE: For handling class imbalance during model training.

Joblib: For saving and loading machine learning models.

Requirements
To run this application, you need the following Python libraries. These can be installed via pip:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt:
streamlit

pandas

numpy

matplotlib

seaborn

plotly

scikit-learn

imbalanced-learn

biopython

joblib

requests

Setup
Clone this repository:

bash
Copy
Edit
git clone https://github.com/yourusername/epitope-prediction.git
cd epitope-prediction
Create and activate a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run UniProt_streamlit_app.py
This will open the application in your default web browser.

How to Use
Data Overview:

View sample datasets for B-cell, T-cell, SARS, and COVID sequences.

Optionally, check the preprocessed training data for B-cell and T-cell prediction.

Model Training:

Select whether you want to train a model for B-cell or T-cell prediction.

The app will train a Random Forest model using the dataset and save the trained model for future predictions.

You can apply SMOTE to handle class imbalance if necessary.

Epitope Prediction:

Enter a protein sequence manually or use a UniProt ID to extract the sequence automatically.

The app will simulate peptide sequences (8-11 amino acids) from the protein sequence.

It will use the trained model to predict which peptides are likely to be epitopes.

A summary of the predicted epitopes will be displayed, along with histograms for different peptide properties (e.g., peptide length, hydrophobicity).

You can download the results as a CSV file.

Model Details

The model is based on Random Forest Classifier and is trained using features such as:

Peptide Length

Hydrophobicity

Stability

Aromaticity

Isoelectric Point

Chou-Fasman Score

Emini Score

Kolaskar-Tongaonkar Score

Parker Score

The training data consists of B-cell and T-cell datasets along with SARS data to help model epitope prediction.

Peptide Generation
The tool generates peptides of lengths between 8 and 11 amino acids from the provided protein sequence.

Each generated peptide is analyzed for various properties using BioPython's ProteinAnalysis class.

Epitope Prediction Workflow
Peptide Generation:

The protein sequence is split into peptides of length 8-11 amino acids.

Features are extracted for each peptide.

Model Prediction:

The pre-trained model is applied to predict whether each peptide is an epitope.

Predictions are displayed along with other peptide properties.

Visualizations:

Histograms and plots for peptide length, hydrophobicity, and other features.

Download a CSV file containing all the peptide data and predictions.

Example

Input Sequence: Paste a protein sequence into the provided text box or enter a UniProt ID to automatically fetch the sequence.

Prediction Result: After predicting, you will see the following:

A table of peptides with their features and prediction results.

Downloadable CSV file containing the data.

Example peptide prediction:

plaintext
Copy
Edit
Peptide: YTFSYGVY
Hydrophobicity: 0.5
Stability: 1.3
Isoelectric Point: 7.0
Prediction: Epitope

 
