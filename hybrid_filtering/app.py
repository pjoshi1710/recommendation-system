from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
import os

# Initialize Flask app
app = Flask(__name__)

# Load data with a relative file path
input_file_path = os.path.join(
    os.path.dirname(__file__), 
    "..", "data", "01_AMAL_Master_Data_PCS.csv"
)

# Verify the file path
print("Current Directory:", os.getcwd())
print("CSV Exists:", os.path.exists(input_file_path))

if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"CSV file not found: {input_file_path}")

# Read the data
df = pd.read_csv(input_file_path)

# Ensure the DataFrame is not empty
if df.empty:
    raise ValueError("Loaded CSV is empty. Please check the data source.")

# Clean up data
df.columns = df.columns.str.strip()
print("Columns after stripping:", df.columns.tolist())

# Required columns
required_columns = [
    "Supplier Name", "Buyer ID", "Buyer Name", "CPV Code Description", 
    "SUPPLIER REGION CODE", "Supplier Country New", 
    "SUPPLIER EMPLOYEE RANGE", "SUPPLIER TURN OVER RANGE"
]

# Ensure all required columns exist
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}")

@app.route('/')
def index():
    """Home page with supplier selection."""
    unique_supplier_names = sorted(set(df['Supplier Name'].dropna().str.strip()))
    return render_template('index.html', suppliers=unique_supplier_names)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    """Generate and display recommendations."""
    selected_supplier = request.form.get('supplier_name')

    if selected_supplier not in df['Supplier Name'].values:
        return f"Invalid Supplier Name: {selected_supplier}", 400

    # Step 1: Get supplier details
    supplier_details = df[df['Supplier Name'] == selected_supplier].iloc[0]
    selected_cpv_desc = supplier_details['CPV Code Description']
    selected_supp_region = supplier_details['SUPPLIER REGION CODE']
    selected_supp_country = supplier_details['Supplier Country New']

    # Step 2: Filter data
    filtered_data = df[
        (df['CPV Code Description'] == selected_cpv_desc) &
        (df['SUPPLIER REGION CODE'] == selected_supp_region) &
        (df['Supplier Country New'] == selected_supp_country)
    ].copy()

    if filtered_data.empty:
        return "No relevant suppliers found for recommendations.", 400

    # Step 3: Handle categorical columns using One-Hot Encoding
    # Example: Convert 'SUPPLIER EMPLOYEE RANGE' to one-hot encoded features
    # One-hot encoding for categorical columns
    encoder = OneHotEncoder(sparse_output=True)
    emp_range_encoded = encoder.fit_transform(filtered_data[['SUPPLIER EMPLOYEE RANGE']])
    turnover_encoded = encoder.fit_transform(filtered_data[['SUPPLIER TURN OVER RANGE']])
    cpv_encoded = encoder.fit_transform(filtered_data[['CPV Code Description']])

    # Calculating cosine similarity
    content_similarity_cpv = cosine_similarity(cpv_encoded)
    content_similarity_emp_range = cosine_similarity(emp_range_encoded)
    content_similarity_turnover = cosine_similarity(turnover_encoded)


    # Step 3.4: Combine the similarities
    # Combine the different similarity matrices into one, ensuring you can blend the similarities appropriately
    # Here we simply average them, but you can give different weights to each similarity based on importance

    # Ensure matrices match filtered data dimensions
    if len(filtered_data) > 0:
        filtered_data['Content Similarity'] = (
            0.4 * content_similarity_cpv[0] +
            0.3 * content_similarity_emp_range[0] +
            0.3 * content_similarity_turnover[0]
        )
    else:
        return "Not enough data to calculate similarity.", 400

    # Step 4: Collaborative Filtering
    interaction_matrix = df.pivot_table(index="Supplier Name", columns="Buyer ID", aggfunc="size", fill_value=0)

    if interaction_matrix.empty:
        return "The interaction matrix is empty. Cannot perform collaborative filtering.", 400

    if selected_supplier not in interaction_matrix.index:
        return f"Supplier '{selected_supplier}' not found in interaction matrix.", 400

    interaction_matrix_sparse = csr_matrix(interaction_matrix)
    collaborative_similarities = cosine_similarity(interaction_matrix_sparse)

    supplier_idx = interaction_matrix.index.get_loc(selected_supplier)
    collaborative_scores = collaborative_similarities[supplier_idx]

    collaborative_scores_df = pd.DataFrame({
        "Supplier Name": interaction_matrix.index,
        "Collaborative Similarity": collaborative_scores
    })

    # Merge collaborative scores into filtered data
    filtered_data = filtered_data.merge(collaborative_scores_df, on="Supplier Name", how="left")
    filtered_data["Collaborative Similarity"] = filtered_data["Collaborative Similarity"].fillna(0)

    # Step 5: Compute Hybrid Score
    content_weight, collaborative_weight = 0.7, 0.3
    filtered_data["Hybrid Score"] = (
        content_weight * filtered_data["Content Similarity"] +
        collaborative_weight * filtered_data["Collaborative Similarity"]
    )

    # Step 6: Generate Final Recommendations
    recommendations = (
        filtered_data.groupby("Supplier Name")
        .agg({
            "SUPPLIER REGION CODE": "first",
            "Supplier Country New": "first",
            "CPV Code Description": "first",
            "SUPPLIER EMPLOYEE RANGE": "first",
            "Buyer ID": "first",
            "Buyer Name": "first",
            "SUPPLIER TURN OVER RANGE": "first",
            "Hybrid Score": "max"
        })
        .reset_index()
        .sort_values(by="Hybrid Score", ascending=False)
    )

    recommendations["Hybrid Score"] = recommendations["Hybrid Score"].fillna(0)

    return render_template(
        "recommendations.html", 
        recommendations=recommendations.to_dict(orient="records"),
        columns=recommendations.columns.tolist()
    )

if __name__ == '__main__':
    app.run(host='10.40.5.221', port=5000, debug=True)

