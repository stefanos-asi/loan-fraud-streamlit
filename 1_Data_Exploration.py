import streamlit as st
import pandas as pd
import time
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import timedelta

# Page configuration
st.set_page_config(page_title="Data Exploration", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Custom sidebar color
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

# Title
st.markdown("<h3 style='text-align: center;'>🔍 Data Exploration & Preprocessing of a Loan Dataset</h3>", unsafe_allow_html=True)

# Datasets info
st.markdown("""
<span style='color:#333;font-family: monospace; font-weight:bold; font-size:16px;'>Two datasets:</span>
<ul style='color:#333; font-weight:bold; font-family: monospace; line-height: 2.4em; font-size:14px;'>
<li>🗂️ Loan applications</li>
<li>🗂️ Transactional data of applicants</li>
</ul>
""", unsafe_allow_html=True)

# Goal
st.markdown("""
<div style='
    border: 2px solid #00c300;
    border-radius: 5px;
    padding: 15px 20px;    
    max-width: 600px;
    background-color: #f6fff6;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    text-align: center;
'>
  <span style='color:#00c300; font-size:15px; font-weight:bold'>
      End Goal: Prediction of fraudulent loans
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# Start button style
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: white;
    color: black;
    font-weight: bold;
    padding: 15px 80px;
    font-size: 16px;
    border: 2px solid black;
    border-radius: 0px;
    transition: background-color 0.3s, color 0.3s;
}
div.stButton > button:first-child:hover {
    background-color: black;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("**Press the button below to load and process the loan and transaction data:**")

# File paths
loans = 'data/loan_applications.csv'
transactions = 'data/transactions.csv'

# Caching functions
@st.cache_data(show_spinner="Loading data...")
def load_data():
    df_loans = pd.read_csv(loans)

    df_transactions = pd.read_csv(transactions)

    return df_loans, df_transactions

@st.cache_data(show_spinner="Processing data...")
def preprocess_data(df_loans, df_transactions):
    df_loans['application_date'] = pd.to_datetime(df_loans['application_date'])
    df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'])
    df_loans['fraud_type'] = df_loans['fraud_type'].fillna('Not Fraudulent')
    return df_loans, df_transactions

@st.cache_data(show_spinner="Generating features...")
def create_transaction_features(df_loans, df_transactions):
    time_windows = [180, 365]
    merged_df = pd.merge(df_loans, df_transactions, how='left', on='customer_id')
    merged_df = merged_df.sort_values(by=['customer_id', 'transaction_date'])

    feature_list = []

    for customer_id, group in merged_df.groupby('customer_id'):
        applications = group.drop_duplicates(subset='application_id')
        for _, app in applications.iterrows():
            app_date = app['application_date']
            app_id = app['application_id']
            past_txns = group[group['transaction_date'] < app_date]
            features = {'application_id': app_id}
            for days in time_windows:
                window_start = app_date - timedelta(days=days)
                window_txns = past_txns[past_txns['transaction_date'] >= window_start]
                features[f'num_of_txns_{days}'] = len(window_txns)
                features[f'total_amt_{days}'] = window_txns['transaction_amount'].sum()
                features[f'avg_amt_{days}'] = window_txns['transaction_amount'].mean() or 0
                features[f'unique_cat_{days}'] = window_txns['merchant_category'].nunique()
            feature_list.append(features)

    features_df = pd.DataFrame(feature_list)

    df_loans = pd.merge(df_loans, features_df, how='left', on='application_id')
    return df_loans

#Cache pandas profile report and creating the function to call it
@st.cache_resource(show_spinner="Generating profiling report..")
def generate_profile_report(df,title):
    return ProfileReport(df,title=title,explorative=True)

# Session state
if 'started' not in st.session_state:
    st.session_state.started = False
if 'show_loan_report' not in st.session_state:
    st.session_state.show_loan_report = False
if 'show_transaction_report' not in st.session_state:
    st.session_state.show_transaction_report = False

# Start button logic
if not st.session_state.started:
    if st.button("Start"):
        st.session_state.started = True
        with st.spinner("🔄 Loading loan and transaction files..."):
            df_loans, df_transactions = load_data()
            df_loans, df_transactions = preprocess_data(df_loans, df_transactions)
            st.session_state.df_loans = df_loans
            st.session_state.df_transactions = df_transactions

# Main app
if st.session_state.started:
    df_loans = st.session_state.df_loans
    df_transactions = st.session_state.df_transactions
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Loan applications data**')
        st.dataframe(df_loans.head(10))
    with col2:
        st.markdown("**Transactions data**")
        st.dataframe(df_transactions.head(10))

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Loan applications table information**")
        with st.expander("📂 Profiling Report"):

            if st.button("▶️ Generate the Loan Report"):
                profile_loans = generate_profile_report(df_loans,"Loan Applications Profiling Report")
                st.session_state.profile_loans = profile_loans
                st.session_state.show_loan_report = True
            if st.session_state.show_loan_report:
                st_profile_report(st.session_state.profile_loans)

    with col2:
        st.markdown("**Transactions table information**")
        with st.expander("📂 Profiling Report"):
            if st.button("▶️ Generate the Transaction Report"):
                profile_trans = generate_profile_report(df_transactions,"Transactions Profiling Report")
                st.session_state.profile_trans = profile_trans
                st.session_state.show_transaction_report = True
            if st.session_state.show_transaction_report:
                st_profile_report(st.session_state.profile_trans)

    st.markdown("---")

    value_counts_fraud_type = df_loans['fraud_type'].value_counts().sort_values(ascending=False).reset_index()
    value_counts_fraud_type.columns = ['Fraud Type after imputation', 'Count']
    st.dataframe(value_counts_fraud_type)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Violin plot of numeric variables after capping on 2nd & 97th percentile:**")
    with col2:
        st.markdown("**Countplot of categorical variables:**")

    col1, col2 = st.columns(2)

    with col1:
        df_loans_copy = df_loans.copy()

        numeric_variables = [col for col in df_loans_copy.select_dtypes(include=np.number) if df_loans_copy[col].nunique() >= 10]

        for col in numeric_variables:
            df_loans_copy[col] = df_loans_copy[col].clip(df_loans_copy[col].quantile(0.02), df_loans_copy[col].quantile(0.97))
        selected_pretty = st.selectbox("", [col.replace("_", " ") for col in numeric_variables])
        selected_var = selected_pretty.replace(" ", "_")
        fig = px.violin(df_loans_copy, y=selected_var, box=True, points=False)
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        categorical_variables = [col for col in df_loans.select_dtypes(include='object') if df_loans[col].nunique() <= 100]
        default_col = 'fraud_type'
        default_index = categorical_variables.index(default_col) if default_col in categorical_variables else 0
        selected_pretty = st.selectbox("", [col.replace("_", " ") for col in categorical_variables], index=default_index)
        selected_var = selected_pretty.replace(" ", "_")
        count_df = df_loans[selected_var].value_counts().reset_index()
        count_df.columns = [selected_pretty, 'Count']
        fig = px.bar(count_df, x=selected_pretty, y='Count')
        fig.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    #feature engineering

    st.markdown("---")


    # Additional features

    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }

    suspicious_purposes = [
        'Wedding',
        'Home Renovation',
        'Business Expansion',
        'Vehicle Purchase'
    ]

    suspicious_purposes = [
        'Wedding',
        'Home Renovation',
        'Business Expansion',
        'Vehicle Purchase'
    ]

    epsilon = 1e-6

    df_loans['age_bins']=pd.cut(df_loans['applicant_age'],bins=[0,25,35,45,70],labels=['-25','26-35','36-45','46-70']).astype(str)

    df_loans['single_provider']=np.where((df_loans['number_of_dependents']>0) & (df_loans['employment_status']=='unemployed'),1,0)

    df_loans['suspicious_activities']=df_loans['purpose_of_loan'].isin([p for p in suspicious_purposes]).astype(int)

    df_loans['dependents_to_income_ratio']=df_loans['number_of_dependents']/df_loans['monthly_income']


    df_loans['loan_to_income_ratio'] = (df_loans['loan_amount_requested'] / (df_loans['monthly_income'] + epsilon)) * 100

    df_loans['existing_debt_to_income'] = (df_loans['existing_emis_monthly'] / (df_loans['monthly_income'] + epsilon)) * 100
    
    df_loans['Year']=df_loans['application_date'].dt.year

    df_loans['Month']=df_loans['application_date'].dt.month

    df_loans['application_season']=df_loans['Month'].map(season_map)

    df_loans['City']=df_loans['residential_address'].apply(lambda x: x.split(',')[-2].strip() if isinstance(x,str) and len(x.split(','))>1 else np.nan)

    df_loans = create_transaction_features(df_loans, df_transactions)

    st.session_state.df_loans_final = df_loans
