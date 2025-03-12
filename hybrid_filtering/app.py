7# Content-Based Filtering (CBF): Recommends suppliers with similar attributes (e.g., type, location, product categories).
# Collaborative Filtering (CF): Recommends suppliers based on user preferences or historical interactions.
# Rule-Based or Business Logic Filtering: Applies domain-specific rules (e.g., location or supplier type constraints).

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import os

# Initialize Flask app
app = Flask(__name__)

# Load data with a relative file path
input_file_path = os.path.join(
    os.path.dirname(__file__), 
    "..", # Move one level up to the parent directory
    "data",  # Add the folder name
    "01_AMAL_Master_Data_PCS.csv"
)

# Read the data
# input_file_path = "C:/Users/Prashant Joshi/OneDrive - elcom.com/Documents/Data Science/Local Spend Analytics/SG/01_AMAL_Master_Data_PCS.csv"
df = pd.read_csv(input_file_path)

# logging.info(f"Columns in the DataFrame: {df.columns.tolist()}")

# Clean up data
df.columns = df.columns.str.strip()
required_columns = ["Supplier Name", "Buyer ID", "Buyer Name", "CPV Code Description", "SUPPLIER REGION CODE", "Supplier Country New", "SUPPLIER EMPLOYEE RANGE", "SUPPLIER TURN OVER RANGE"]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_columns)

@app.route('/')
def index():
    """Home page with supplier selection."""
    # unique_supplier_names = df['Supplier Name'].dropna().str.strip().drop_duplicates().sort_values()


    # Get unique supplier names, ensuring proper formatting
    supplier_names = df['Supplier Name'].dropna().str.strip().str.lower().unique()

# Convert back to original case for display
    unique_supplier_names = sorted(set(df['Supplier Name'].dropna().str.strip()))
    return render_template('index.html', suppliers=unique_supplier_names)



@app.route('/recommendations', methods=['POST'])
def recommendations():
    """Generate and display recommendations."""
    selected_supplier = request.form.get('supplier_name')
    
    if selected_supplier not in df['Supplier Name'].values:
        return "Invalid Supplier Name", 400

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
    encoder = OneHotEncoder(sparse=True)
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

    # Step 4: Collaborative filtering
    # interaction matrix shows the frequency (or count) of interactions between each supplier and buyer.
    interaction_matrix = df.pivot_table(index='Supplier Name', columns='Buyer ID', aggfunc='size', fill_value=0)
    if interaction_matrix.empty:
        return "The interaction matrix is empty. Cannot perform collaborative filtering.", 400

    if selected_supplier not in interaction_matrix.index:
        return f"Supplier '{selected_supplier}' not found in interaction matrix.", 400

    # Convert to sparse matrix and compute similarities
    interaction_matrix_sparse = csr_matrix(interaction_matrix)
    collaborative_similarities = cosine_similarity(interaction_matrix_sparse)

    # Find the index of the selected supplier
    supplier_idx = interaction_matrix.index.get_loc(selected_supplier)
    collaborative_scores = collaborative_similarities[supplier_idx]

    # Create collaborative similarity DataFrame
    collaborative_scores_df = pd.DataFrame({
        'Supplier Name': interaction_matrix.index,
        'Collaborative Similarity': collaborative_scores
    })

    # Merge collaborative scores into filtered data
    filtered_data = filtered_data.merge(collaborative_scores_df, on='Supplier Name', how='left')
    filtered_data['Collaborative Similarity'] = filtered_data['Collaborative Similarity'].fillna(0)

    # Step 5: Hybrid score
    content_weight = 0.7
    collaborative_weight = 0.3
    filtered_data['Hybrid Score'] = (
        content_weight * filtered_data['Content Similarity'] +
        collaborative_weight * filtered_data['Collaborative Similarity']
    )

    # Step 6: Final recommendations
    recommendations = (
        filtered_data.groupby('Supplier Name')
        .agg({
            'SUPPLIER REGION CODE': 'first',
            'Supplier Country New': 'first',
            'CPV Code Description': 'first',
            'SUPPLIER EMPLOYEE RANGE': 'first',
            'Buyer ID': 'first',
            'Buyer Name': 'first',
            'SUPPLIER TURN OVER RANGE': 'first',
            'Hybrid Score': 'max'
        })
        .reset_index()
        .sort_values(by='Hybrid Score', ascending=False)
    )

    # Handle null values in Hybrid Score
    recommendations['Hybrid Score'] = recommendations['Hybrid Score'].fillna(0)

    # Debug recommendations
    # print("Final Recommendations:")
    # print(recommendations.head())

    # Pass recommendations to the template
    return render_template(
        'recommendations.html', 
        recommendations=recommendations.to_dict(orient='records'),
        columns=recommendations.columns.tolist()
    )


if __name__ == '__main__':
    app.run(debug=True)
