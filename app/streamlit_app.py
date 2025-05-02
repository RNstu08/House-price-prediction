import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration ---
# Determine base path relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models') # Go up one level, then into models

MODEL_PATH = os.path.join(MODEL_DIR, 'house_price_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'feature_columns.pkl')

# --- Load Artifacts ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(COLUMNS_PATH, 'rb') as f:
        feature_columns = pickle.load(f)
    # print("Model, scaler, and columns loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading model artifacts: {e}")
    st.error("Please ensure 'house_price_model.pkl', 'scaler.pkl', and 'feature_columns.pkl' exist in the 'models/' directory.")
    st.stop() # Stop the app if artifacts aren't found
except Exception as e:
    st.error(f"An error occurred during artifact loading: {e}")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏡 House Price Prediction App")
st.write("Enter the details of the house to get a price prediction.")

# --- Input Features ---
# IMPORTANT: You need input fields for features used by your model.
# Creating inputs for ALL ~200+ one-hot encoded features is impractical.
# Strategies:
# 1. Select a SUBSET of the MOST IMPORTANT original features for the UI.
# 2. Create the *full* feature set programmatically based on these inputs.
# 3. Handle categorical inputs carefully (map to numerical if needed before OHE).

st.sidebar.header("Input Features (Example Subset)")

# Example Inputs (Replace/add based on important features from your analysis)
# Use reasonable defaults (e.g., median values from training set)
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
total_bsmt_sf = st.sidebar.number_input("Basement Area (sq ft)", min_value=0, max_value=6000, value=850)
st_flr_sf = st.sidebar.number_input("First Floor Area (sq ft)", min_value=300, max_value=5000, value=1100)
nd_flr_sf = st.sidebar.number_input("Second Floor Area (sq ft)", min_value=0, max_value=3000, value=0)
garage_cars = st.sidebar.slider("Garage Capacity (Cars)", 0, 5, 2)
garage_area = st.sidebar.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=480)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=2005)
year_remod_add = st.sidebar.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2005)
yr_sold = 2025 # Assume current year for prediction, or make it an input

# Example Categorical Input (Needs careful handling for encoding)
neighborhood = st.sidebar.selectbox("Neighborhood", ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Other']) # Example list


# --- Feature Engineering & Preprocessing for Input ---
# This part is CRITICAL and must perfectly mirror the notebook preprocessing

st.header("Processing Input...")

# Create a dictionary to hold input data
input_data = {}

# --- Step 1: Start with features directly from input ---
input_data['OverallQual'] = overall_qual
input_data['GrLivArea'] = gr_liv_area
input_data['TotalBsmtSF'] = total_bsmt_sf
input_data['1stFlrSF'] = st_flr_sf
input_data['2ndFlrSF'] = nd_flr_sf
input_data['GarageCars'] = garage_cars
input_data['GarageArea'] = garage_area
input_data['YearBuilt'] = year_built
input_data['YearRemodAdd'] = year_remod_add
input_data['YrSold'] = yr_sold
# ... add ALL other raw features your model *indirectly* depends on (even if not direct inputs)
input_data['FullBath'] = st.sidebar.slider("Full Bathrooms Above Grade", 0, 4, 1) # Example Add
input_data['HalfBath'] = st.sidebar.slider("Half Bathrooms Above Grade", 0, 2, 0) # Example Add
input_data['BsmtFullBath'] = st.sidebar.slider("Basement Full Bathrooms", 0, 3, 0) # Example Add
input_data['BsmtHalfBath'] = st.sidebar.slider("Basement Half Bathrooms", 0, 2, 0) # Example Add
input_data['Fireplaces'] = st.sidebar.slider("Number of Fireplaces", 0, 4, 0) # Example Add
input_data['PoolArea'] = st.sidebar.number_input("Pool Area (sq ft)", min_value=0, max_value=1000, value=0) # Example Add
input_data['OpenPorchSF'] = st.sidebar.number_input("Open Porch Area (sq ft)", min_value=0, max_value=800, value=45) # Example Add
input_data['EnclosedPorch'] = st.sidebar.number_input("Enclosed Porch Area (sq ft)", min_value=0, max_value=600, value=0) # Example Add
input_data['3SsnPorch'] = st.sidebar.number_input("3-Season Porch Area (sq ft)", min_value=0, max_value=600, value=0) # Example Add
input_data['ScreenPorch'] = st.sidebar.number_input("Screen Porch Area (sq ft)", min_value=0, max_value=600, value=0) # Example Add

# --- Step 2: Derive engineered features ---
input_data['TotalSF'] = input_data['TotalBsmtSF'] + input_data['1stFlrSF'] + input_data['2ndFlrSF']
input_data['HouseAge'] = input_data['YrSold'] - input_data['YearBuilt']
input_data['RemodAge'] = input_data['YrSold'] - input_data['YearRemodAdd']
input_data['IsRemodeled'] = int(input_data['YearRemodAdd'] != input_data['YearBuilt'])
# ... add ALL other engineered features created in the notebook
# --- Additional Engineered Features (Examples) ---

# Total number of bathrooms (half baths count as 0.5)
input_data['TotalBath'] = (input_data['FullBath'] + 0.5 * input_data['HalfBath'] +
                           input_data['BsmtFullBath'] + 0.5 * input_data['BsmtHalfBath'])

# Total porch square footage
input_data['TotalPorchSF'] = (input_data['OpenPorchSF'] + input_data['EnclosedPorch'] +
                              input_data['3SsnPorch'] + input_data['ScreenPorch'])

# Boolean flags for amenities (based on numerical features)
input_data['HasPool'] = int(input_data['PoolArea'] > 0)
input_data['HasGarage'] = int(input_data['GarageArea'] > 0)
input_data['HasBsmt'] = int(input_data['TotalBsmtSF'] > 0)
input_data['Has2ndFlr'] = int(input_data['2ndFlrSF'] > 0)
input_data['HasFireplace'] = int(input_data['Fireplaces'] > 0)

# Example Ratio Feature (handle division by zero)
if input_data['GarageCars'] > 0:
    input_data['GarageAreaPerCar'] = input_data['GarageArea'] / input_data['GarageCars']
else:
    input_data['GarageAreaPerCar'] = 0

# Example Quality Interaction (consider if this was useful in your model)
# input_data['OverallGrade'] = input_data['OverallQual'] * input_data['OverallCond']

# Example Polynomial Feature (if OverallQual was important and showed non-linearity)
# input_data['OverallQual_sq'] = input_data['OverallQual'] ** 2

# --- End of Additional Engineered Features ---

# --- Step 3: Handle categorical inputs ---
# This is complex. We need to create the one-hot encoded columns.
# Example for Neighborhood:
neighborhood_cols = [f'Neighborhood_{n}' for n in ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst']] # Match notebook OHE
for col in neighborhood_cols:
    input_data[col] = 0 # Initialize all related OHE columns to 0
if neighborhood != 'Other':
    target_neigh_col = f'Neighborhood_{neighborhood}'
    if target_neigh_col in feature_columns: # Check if this column exists
         input_data[target_neigh_col] = 1
# ---> You MUST replicate this logic for ALL categorical features used in the model <---
# ---> This is where a preprocessing function/pipeline from `utils.py` is very helpful! <---

# --- Step 4: Create DataFrame with correct columns ---
# Create a DataFrame with a single row, containing zeros for all expected feature columns
input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0 # Initialize row with zeros

# Fill in the values from input_data for matching columns
for key, value in input_data.items():
    if key in input_df.columns:
        # Apply log1p if this feature was transformed in the notebook
        if key in ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'TotalSF', 'GarageArea']: # Example list of skewed features
             input_df.loc[0, key] = np.log1p(value)
        else:
             input_df.loc[0, key] = value
    # else:
    #     st.warning(f"Input key '{key}' not found in model feature columns.") # Debugging

# Ensure all columns required by the model exist, fill missing with 0 or appropriate default
# This step should ideally be minimal if input_data preparation is correct.
input_df = input_df.reindex(columns=feature_columns, fill_value=0)


st.subheader("Input Data Overview (Processed)")
st.dataframe(input_df.head()) # Show the final DataFrame row being used

# --- Prediction ---
if st.button("Predict House Price"):
    try:
        # --- Step 5: Scale the input data ---
        input_scaled = scaler.transform(input_df) # Use the loaded scaler
        st.write("Input scaled successfully.")

        # --- Step 6: Make Prediction ---
        prediction_log = model.predict(input_scaled)
        st.write("Prediction (log scale) successful.")

        # --- Step 7: Inverse Transform Prediction ---
        # Since we predicted log(1+price), we need exp(prediction) - 1
        prediction_final = np.expm1(prediction_log[0])

        st.subheader("Prediction Result")
        st.success(f"Predicted House Price: ${prediction_final:,.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check the input values and ensure the model artifacts are correct.")
        st.dataframe(input_df) # Show df for debugging