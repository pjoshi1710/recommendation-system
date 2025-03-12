from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Clean up data
df.columns = df.columns.str.strip()
required_columns = ["Supplier Name", "Buyer ID", "Buyer Name", "CPV Code Description", "SUPPLIER REGION CODE", "Supplier Country New", "SUPPLIER EMPLOYEE RANGE", "SUPPLIER TURN OVER RANGE"]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_columns)



@app.route('/')
def index():
    """Home page with supplier selection."""
    unique_supplier_names = df['Supplier Name'].dropna().str.strip().drop_duplicates().sort_values()
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

    if len(filtered_data) < 2:
        return "Not enough data to generate recommendations.", 400

    # Step 3: Calculate content similarity
    vectorizer = TfidfVectorizer()
    cpv_vectors = vectorizer.fit_transform(filtered_data['CPV Code Description'].astype(str))

    # Calculate similarity
    content_similarities = cosine_similarity(cpv_vectors)
    filtered_data.loc[:, 'Similarity Score'] = content_similarities[0]


    # Step 4: Final recommendations
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
            'Similarity Score': 'max'
        })
        .reset_index()
        .sort_values(by='Similarity Score', ascending=False)
    )

    # Debug recommendations
    print("Final Recommendations:")
    print(recommendations.head())

    # Pass recommendations to the template
    return render_template(
    'recommendations.html', 
    recommendations=recommendations.to_dict(orient='records'),
    columns=recommendations.columns.tolist()
)


if __name__ == '__main__':
    app.run(debug=True)
