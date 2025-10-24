# --- Necessary Imports ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os
import joblib # Using joblib might be better for caching complex scikit-learn objects

# --- Configuration ---
DATA_FILE = 'CC GENERAL.csv'
N_COMPONENTS_PCA = 8
OPTIMAL_K = 3 # Set this based on your previous analysis or make it configurable

# --- Load Data ---
@st.cache_data # Cache the raw data loading
def load_data(file_path=DATA_FILE):
    """Loads training data from the specified CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found.")
        return None
    try:
        df = pd.read_csv(file_path)
        if 'CUST_ID' in df.columns:
            df = df.drop('CUST_ID', axis=1)
        st.success(f"Successfully loaded training data from '{file_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {e}")
        return None

# --- Train Preprocessor ---
@st.cache_resource # Cache the fitted preprocessor
def train_preprocessor(_df_train):
    """Fits and returns the preprocessing pipeline."""
    df_process = _df_train.copy()

    # Identify column types automatically
    numerical_cols_all = df_process.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_process.select_dtypes(exclude=np.number).columns.tolist()

    # Define numerical columns needing specific imputation
    num_cols_median_impute = ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']
    numerical_cols_for_scaling = [col for col in numerical_cols_all if col not in num_cols_median_impute]

    # --- Create Preprocessing Pipelines ---
    numerical_pipeline_impute_median = Pipeline(steps=[
        ('imputer_median', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    numerical_pipeline_scale_only = Pipeline(steps=[
        ('imputer_mean', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    transformers_list = [
        ('num_scale_only', numerical_pipeline_scale_only, numerical_cols_for_scaling),
        ('num_impute_median', numerical_pipeline_impute_median, num_cols_median_impute)
    ]

    # Add categorical pipeline if needed
    if categorical_cols:
        categorical_pipeline = Pipeline(steps=[
            ('imputer_mode', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        transformers_list.append(('cat', categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='passthrough'
    )

    try:
        preprocessor.fit(df_process)
        st.write("✅ Preprocessing pipeline trained.")
        # Store column names used for training
        st.session_state['training_columns'] = df_process.columns.tolist()
        return preprocessor
    except Exception as e:
        st.error(f"Error training preprocessor: {e}")
        return None

# --- Train PCA ---
@st.cache_resource # Cache the fitted PCA model
def train_pca(_X_processed, n_components=N_COMPONENTS_PCA):
    """Fits and returns the PCA model."""
    actual_n_components = n_components
    if _X_processed.shape[1] < n_components:
        actual_n_components = _X_processed.shape[1]
        st.warning(f"Adjusting PCA components to {actual_n_components} due to fewer input features.")

    pca = PCA(n_components=actual_n_components)
    try:
        pca.fit(_X_processed)
        st.write(f"✅ PCA model trained (n_components={actual_n_components}).")
        st.write(f"   Explained Variance Ratio: {pca.explained_variance_ratio_.sum():.3f}")
        return pca
    except Exception as e:
        st.error(f"Error training PCA: {e}")
        return None

# --- Train K-Means ---
@st.cache_resource # Cache the fitted K-Means model
def train_kmeans(_X_pca, k=OPTIMAL_K):
    """Fits and returns the K-Means model."""
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    try:
        kmeans.fit(_X_pca)
        st.write(f"✅ K-Means model trained (k={k}).")
        return kmeans
    except Exception as e:
        st.error(f"Error training K-Means: {e}")
        return None

# --- Define Cluster Names ---
# IMPORTANT: Verify this mapping based on your cluster analysis!
cluster_name_map = {
    0: "Premium Customers (High spending)",
    1: "Low-Spend Customers",
    2: "Credit-Hungry Customers"
}
# Adjust if OPTIMAL_K is not 3 or labels are different
if OPTIMAL_K != 3:
     cluster_name_map = {i: f"Cluster {i+1}" for i in range(OPTIMAL_K)}


# === Streamlit App Layout ===
st.set_page_config(layout="wide")
st.title("Customer Cluster Prediction")

# --- Model Training Section (runs once and caches) ---
with st.spinner("Loading data and training models... This might take a moment on first run."):
    df_train = load_data()
    if df_train is not None:
        # Check if models are already cached
        if 'preprocessor' not in st.session_state:
             st.session_state['preprocessor'] = train_preprocessor(df_train)

        if st.session_state['preprocessor'] is not None and 'pca_model' not in st.session_state:
             X_processed_train = st.session_state['preprocessor'].transform(df_train) # Use transform here
             st.session_state['pca_model'] = train_pca(X_processed_train)

        if st.session_state['pca_model'] is not None and 'kmeans_model' not in st.session_state:
             X_pca_train = st.session_state['pca_model'].transform(X_processed_train)
             st.session_state['kmeans_model'] = train_kmeans(X_pca_train)
    else:
        st.stop() # Stop if data loading failed

# Check if all models trained successfully
if 'preprocessor' not in st.session_state or st.session_state['preprocessor'] is None or \
   'pca_model' not in st.session_state or st.session_state['pca_model'] is None or \
   'kmeans_model' not in st.session_state or st.session_state['kmeans_model'] is None:
    st.error("Model training failed. Please check the data and code.")
    st.stop()


st.success("Models trained successfully!")
st.markdown("---")


# --- User Input Section ---
st.header("Enter New Customer Data")

# Get columns used during training
if 'training_columns' not in st.session_state:
     st.error("Training columns not found in session state.")
     st.stop()

training_cols = st.session_state['training_columns']
input_data = {}

# Create input fields dynamically based on training columns
cols = st.columns(3) # Arrange inputs in columns
col_index = 0
for feature in training_cols:
    with cols[col_index % 3]:
        # Use number_input for numerical, text_input or selectbox for categorical
        # Heuristic: Check dtype from original training df
        if pd.api.types.is_numeric_dtype(df_train[feature].dtype):
            min_val = float(df_train[feature].min())
            max_val = float(df_train[feature].max())
            mean_val = float(df_train[feature].mean())
            # Handle potential NaN mean
            default_val = mean_val if not np.isnan(mean_val) else min_val
            # Ensure default is within bounds
            default_val = max(min_val, min(max_val, default_val))
            input_data[feature] = st.number_input(
                f"Enter {feature}",
                min_value=min_val - (abs(min_val)*0.5), # Allow some range outside training min/max
                max_value=max_val + (abs(max_val)*0.5),
                value=default_val, # Default to mean or min
                step=0.1 if pd.api.types.is_float_dtype(df_train[feature].dtype) else 1.0
            )
        else: # Assuming categorical (object/string type)
             options = df_train[feature].unique().tolist()
             # Handle potential NaNs in options if SimpleImputer wasn't perfect
             options = [opt for opt in options if pd.notna(opt)]
             default_option = options[0] if options else ""
             input_data[feature] = st.selectbox(
                 f"Select {feature}",
                 options=options,
                 index = 0 # Default to first option
             )
    col_index += 1


# --- Prediction ---
if st.button("Predict Cluster"):
    # Convert input data to DataFrame in the correct order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[training_cols] # Ensure column order matches training

    st.subheader("Prediction Result")
    st.write("Input Data:")
    st.dataframe(input_df)

    try:
        # 1. Preprocess the input data using the *fitted* preprocessor
        input_processed = st.session_state['preprocessor'].transform(input_df)

        # 2. Transform using the *fitted* PCA model
        input_pca = st.session_state['pca_model'].transform(input_processed)

        # 3. Predict using the *fitted* K-Means model
        predicted_cluster_label = st.session_state['kmeans_model'].predict(input_pca)[0]

        # 4. Map label to name
        predicted_cluster_name = cluster_name_map.get(predicted_cluster_label, f"Unknown Cluster ({predicted_cluster_label})")

        st.success(f"The customer belongs to Cluster: **{predicted_cluster_name}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please ensure the input data format matches the training data.")

