import streamlit as st
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Single Loan Prediction", layout="wide")

st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #2f2f2f;
        color: white;
    }
    section[data-testid="stSidebar"] * {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center;'>Predict if a loan is fraud</h3>", unsafe_allow_html=True)

# --- Validation ---
if 'df_loans_final' not in st.session_state or 'models' not in st.session_state or 'preprocessor' not in st.session_state:
    st.warning("Please run Page 2 first to train the models.")
    st.stop()

# --- Load from session ---
df = st.session_state.df_loans_final.copy()

models = st.session_state.models

preprocessor = st.session_state.preprocessor

# --- Prepare Input UI ---

x_columns_to_keep = [
    'loan_amount_requested', 'loan_tenure_months', 'interest_rate_offered', 'monthly_income',
    'cibil_score', 'existing_emis_monthly', 'debt_to_income_ratio', 'applicant_age','loan_to_income_ratio',
    'number_of_dependents', 'loan_type', 'purpose_of_loan', 'employment_status',
    'property_ownership_status', 'gender', 'Month', 'Year', 'num_of_txns_180',
    'total_amt_180', 'avg_amt_180', 'unique_cat_180', 'num_of_txns_365',
    'total_amt_365', 'avg_amt_365', 'unique_cat_365','City','age_bins','single_provider','suspicious_activities','application_season','dependents_to_income_ratio'
]

#Columns categorization & user form preparation

X = df[x_columns_to_keep]

num_cols = X.select_dtypes(include=np.number).columns.tolist()

binary_cols=['single_provider','suspicious_activities']

num_cols=[col for col in num_cols if col not in binary_cols]

cat_cols = X.select_dtypes(include='object').columns.tolist()

all_cols=X.columns.tolist()

all_cols=[col for col in all_cols if col!='age_bins']

column_rename_map = {
    'num_of_txns_180':'Number of transactions 180 days',

    'total_amt_180':'Total amount 180 days',
    'unique_cat_180':'Unique categories 180',
    'unique_cat_365':'Unique categories 365',
    'num_of_txns_365':'Number of transactions 365 days',
    'avg_amt_365':'Average amount 365 days',
    'total_amt_365':'Total Amount 365 days',
    'avg_amt_180':'Average Amount 180 days'}

    #Form start

# Age bin function

def get_age_bin(age):
    if age <= 25:
        return '-25'
    elif age <= 35:
        return '26-35'
    elif age <= 45:
        return '36-45'
    elif age <= 70:
        return '46-70'

# Form start
with st.form("user_input_form"):

    user_input = {}

    col_left, col_right = st.columns(2)

    for i, col in enumerate(all_cols):
        col_df = X[col]
        target_col = col_left if i % 2 == 0 else col_right

        if np.issubdtype(col_df.dtype, np.number):
            col_mean = round(col_df.mean(), 2 if col == 'interest_rate_offered' else 0)
            col_min = round(col_df.min(), 0)
            col_max = round(col_df.max(), 0)
            step_val = 0.01 if col == 'interest_rate_offered' else 1

            user_input[col] = target_col.number_input(
                f"{column_rename_map.get(col, col.replace('_', ' ').title())} (Min: {col_min}, Max: {col_max})",
                min_value=int(col_min),
                max_value=int(col_max),
                value=int(col_mean),
                step=int(step_val)
            )
        else:
            options = sorted(col_df.dropna().unique())
            default_val = col_df.mode()[0]        
            user_input[col] = target_col.selectbox(
                f"{column_rename_map.get(col, col.replace('_', ' ').title())}",
                options,
                index=options.index(default_val)
            )

    user_input['age_bins'] = get_age_bin(user_input['applicant_age'])

    input_df= pd.DataFrame([user_input])

    submit = st.form_submit_button('Predict')


if submit:

    #Preprocessing

    input_processed=preprocessor.transform(input_df)

    if hasattr(input_processed,"toarray"):

        input_dense=input_processed.toarray()

    else:

        input_dense=input_processed

    iso=st.session_state.isolation_forest

    input_score=iso.decision_function(input_dense).reshape(-1,1)

    input_with_score=np.hstack([input_dense,input_score])

    #Model

    lgbm_model=models.get('LightGBM')

    if lgbm_model is None:

        st.error('No model found to run the data')

    else:
        #Predict

        pred_proba=lgbm_model.predict_proba(input_with_score)[0][1]

        pred_proba_not_fraud=1-pred_proba

        pred_class=int(pred_proba>=0.57)

        #Display
        st.markdown('---')

        if pred_class==1:
            st.error(f'This loan is predicted as fraud with probability of {pred_proba:.2%}')

        else:
            st.success(f'This loan is predicted as not fraud with a probability of {pred_proba_not_fraud:.2%}')





 