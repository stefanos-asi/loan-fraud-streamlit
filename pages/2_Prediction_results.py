import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.combine import SMOTEENN
from scipy.sparse import csr_matrix
import plotly.figure_factory as ff
from sklearn.model_selection import GridSearchCV
import os
import joblib

# --- Page Config ---
st.set_page_config(page_title="Loan Fraud Prediction", layout="wide")

# Sidebar Styling
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
st.markdown("<h3 style='text-align: center;'>🔍 Prediction of Fraudulent Loans</h3>", unsafe_allow_html=True)

# Additional Styling
st.markdown("""
    <style>
    .metric-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 115px;
        background-color: #fff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        text-align: center;
        font-family: monospace;
    }
    .metric-label {
        font-size: 15px;
        font-weight:bold;
        color: #bbb;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 20px;
        font-family:monospace;
        font-weight: bold;
        color: #4cd137;
    }
    .custom-table {
        font-family: monospace;
        font-size: 14px;
        border-collapse: collapse;
        width: 100%;
        text-align: center;
    }
    .custom-table th, .custom-table td {
        padding: 8px;
        border: 1px solid #ccc;
    }
    .custom-table thead {
        background-color: #f2f2f2;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---

MODEL_DIR='saved_models'
os.makedirs(MODEL_DIR,exist_ok=True)

def save_model(model,name):
    joblib.dump(model,os.path.join(MODEL_DIR,f"{name}.pkl"))

def load_model(name):
    path=os.path.join(MODEL_DIR,f'{name}.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

if 'df_loans_final' not in st.session_state:
    st.warning("Run preprocessing page first")
    st.stop()

df_loans_final = st.session_state.df_loans_final

df_loans_final[df_loans_final.columns[-8:]] = df_loans_final[df_loans_final.columns[-8:]].fillna(0)

x_columns_to_keep = [
    'loan_amount_requested', 'loan_tenure_months', 'interest_rate_offered', 'monthly_income',
    'cibil_score', 'existing_emis_monthly', 'debt_to_income_ratio', 'applicant_age','loan_to_income_ratio',
    'number_of_dependents', 'loan_type', 'purpose_of_loan', 'employment_status',
    'property_ownership_status', 'gender', 'Month', 'Year', 'num_of_txns_180',
    'total_amt_180', 'avg_amt_180', 'unique_cat_180', 'num_of_txns_365',
    'total_amt_365', 'avg_amt_365', 'unique_cat_365','City','age_bins','single_provider','suspicious_activities','application_season','dependents_to_income_ratio'
]

X = df_loans_final[x_columns_to_keep]


y = df_loans_final['fraud_flag']

@st.cache_resource
def get_data():
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True,drop=None), cat_cols)
    ], remainder='passthrough')

    X_processed = preprocessor.fit_transform(X)
    if not isinstance(X_processed, csr_matrix):
        X_processed = csr_matrix(X_processed)

    # Split first
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)

    X_train_dense = X_train.toarray()

    X_test_dense = X_test.toarray()

    # --- Fit Isolation Forest on training data ---
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

    iso.fit(X_train_dense)

    train_scores = iso.decision_function(X_train_dense).reshape(-1, 1)

    test_scores = iso.decision_function(X_test_dense).reshape(-1, 1)

    X_train_with_score = np.hstack([X_train_dense, train_scores])

    X_test_with_score = np.hstack([X_test_dense, test_scores])

    st.session_state.isolation_forest=iso

    # Apply SMOTE
    smote = SMOTEENN(random_state=42,sampling_strategy=0.2)

    X_train_sm, y_train_sm = smote.fit_resample(X_train_with_score, y_train)

    X_train_sm = csr_matrix(X_train_sm)

    return X_train_sm, csr_matrix(X_test_with_score), y_train_sm, y_test, preprocessor


X_train_sm, X_test, y_train_sm, y_test, preprocessor = get_data()

X_train_data = (X_train_sm.data, X_train_sm.indices, X_train_sm.indptr, X_train_sm.shape)


def train_model(model_name, X_train_data, y_train):
    X_train = csr_matrix((X_train_data[0], X_train_data[1], X_train_data[2]), shape=X_train_data[3])

    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=2000)
        model.fit(X_train, y_train)
        return model

    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model

    elif model_name == "LightGBM":

            X_train_dense = X_train.toarray()

            param_grid = {
                'num_leaves': [15, 31],
                'max_depth': [-1, 5],
                'learning_rate': [0.01, 0.05],
                'n_estimators': [100, 300, 500],
                'scale_pos_weight': [10,15,18],
                'class_weight': ['balanced']
            }

            base_model = LGBMClassifier(random_state=42)

            grid = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X_train_dense, y_train)

            best_model = grid.best_estimator_
            return best_model

    else:
        raise ValueError(f"Unknown model_name {model_name}")

def load_or_train(model_name,X_train_data,y_train):
    model=load_model(model_name)
    if model is not None:
        return model
    model=train_model(model_name,X_train_data,y_train)
    save_model(model,model_name)
    return model


# Train and store models
models = {
    "Logistic Regression": load_or_train("Logistic Regression", X_train_data, y_train_sm),

    "Random Forest": load_or_train("Random Forest", X_train_data, y_train_sm),

    "LightGBM": load_or_train("LightGBM", X_train_data, y_train_sm)
}

st.session_state.models = models
st.session_state.preprocessor = preprocessor

def render_model_section(title, model_name, fig_key):

    st.markdown(f"**{title}**")

    model = models[model_name]


    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= 0.57).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    fp_pct = fp / total
    fn_pct = fn / total

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">ROC AUC Score</div>
            <div class="metric-value">{roc_auc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">FP % of total</div>
            <div class="metric-value">{fp_pct:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">FN % of total</div>
            <div class="metric-value">{fn_pct:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T.round(2)
    report_df.rename(index={"0": "Not Fraud", "1": "Fraud"}, inplace=True)
    table_html = report_df.to_html(classes="custom-table", border=0)
    styled_html = f"""
    <div class="metric-container" style="height:auto; padding: 10px;">
        <div class="metric-label" style="color:#bbb; font-size:14px;font-weight:bold">Classification Report</div>
        {table_html}
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)

    st.markdown("---")
    labels = ['Not Fraud', 'Fraud']
    z_text = [[str(cell) for cell in row] for row in cm]
    fig = ff.create_annotated_heatmap(cm, x=labels, y=labels, annotation_text=z_text, colorscale='Blues')
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", margin=dict(t=30, l=50))
    st.plotly_chart(fig, use_container_width=True, key=fig_key)

col1, col2, col3 = st.columns(3)
with col1:
    render_model_section("Logistic Regression", "Logistic Regression", fig_key="logreg_cm")
with col2:
    render_model_section("Random Forest", "Random Forest", fig_key="rf_cm")
with col3:
    render_model_section("LightGBM", "LightGBM", fig_key="lgbm_cm") 
