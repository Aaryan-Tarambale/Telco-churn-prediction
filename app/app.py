
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ── Page config
st.set_page_config(
    page_title='Telco Churn Predictor',
    page_icon='📡',
    layout='centered'
)
# ── Load model
with open('app/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
# ── Header
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>📡 Telco Churn Predictor</h1>
    <p style='text-align: center; color: grey;'>
        Predict whether a customer is likely to leave based on their profile.
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Input form
with st.form('prediction_form'):

    st.markdown("### 📋 Contract & Billing")
    col1, col2, col3 = st.columns(3)
    with col1:
        Contract         = st.selectbox('Contract Type',     ['Month-to-month', 'One year', 'Two year'])
    with col2:
        PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
    with col3:
        PaymentMethod    = st.selectbox('Payment Method',    ['Electronic check', 'Mailed check',
                                                               'Bank transfer (automatic)',
                                                               'Credit card (automatic)'])

    st.markdown("### 💰 Account Info")
    col4, col5, col6 = st.columns(3)
    with col4:
        tenure         = st.slider('Tenure (months)', 0, 72, 12)
    with col5:
        MonthlyCharges = st.number_input('Monthly Charges ($)', 0.0, 150.0, 65.0)
    with col6:
        TotalCharges   = st.number_input('Total Charges ($)', 0.0, 9000.0, 500.0)

    st.markdown("### 🌐 Internet & Security")
    col7, col8, col9 = st.columns(3)
    with col7:
        InternetService  = st.selectbox('Internet Service',  ['Fiber optic', 'DSL', 'No'])
    with col8:
        OnlineSecurity   = st.selectbox('Online Security',   ['Yes', 'No', 'No internet service'])
    with col9:
        TechSupport      = st.selectbox('Tech Support',      ['Yes', 'No', 'No internet service'])

    st.markdown("### 📺 Streaming")
    col10, col11 = st.columns(2)
    with col10:
        StreamingTV      = st.selectbox('Streaming TV',      ['Yes', 'No', 'No internet service'])
    with col11:
        StreamingMovies  = st.selectbox('Streaming Movies',  ['Yes', 'No', 'No internet service'])

    st.markdown("### 👤 Personal")
    col12, col13, col14 = st.columns(3)
    with col12:
        Dependents    = st.selectbox('Dependents',    ['Yes', 'No'])
    with col13:
        MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    with col14:
        OnlineBackup  = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])

    submitted = st.form_submit_button('🔍 Predict Churn', use_container_width=True)

# ── On submit
if submitted:

    # feature engineering
    tenure_group = pd.cut([tenure], bins=[0, 12, 24, 48, 72],
                           labels=['new', 'mid', 'long_term', 'loyal'])[0]

    if StreamingTV == 'Yes' and StreamingMovies == 'Yes':
        subs_type = 'premium'
    elif StreamingTV == 'Yes' and StreamingMovies == 'No':
        subs_type = 'regular'
    elif StreamingTV == 'No' and StreamingMovies == 'Yes':
        subs_type = 'advanced'
    elif StreamingTV == 'No internet service' or StreamingMovies == 'No internet service':
        subs_type = 'no_internet'
    else:
        subs_type = 'none'

    avg_monthly_charges    = TotalCharges / (tenure + 1)
    monthly_to_total_ratio = MonthlyCharges / (TotalCharges + 1)

    # defaults for non-important features
    input_df = pd.DataFrame([{
        'tenure':                 tenure,
        'MonthlyCharges':         MonthlyCharges,
        'TotalCharges':           TotalCharges,
        'avg_monthly_charges':    avg_monthly_charges,
        'monthly_to_total_ratio': monthly_to_total_ratio,
        'gender':                 'Male',
        'Partner':                'No',
        'Dependents':             Dependents,
        'PhoneService':           'Yes',
        'MultipleLines':          MultipleLines,
        'InternetService':        InternetService,
        'OnlineSecurity':         OnlineSecurity,
        'OnlineBackup':           OnlineBackup,
        'DeviceProtection':       'No',
        'TechSupport':            TechSupport,
        'StreamingTV':            StreamingTV,
        'StreamingMovies':        StreamingMovies,
        'Contract':               Contract,
        'PaperlessBilling':       PaperlessBilling,
        'PaymentMethod':          PaymentMethod,
        'tenure_group':           str(tenure_group),
        'subs_type':              subs_type
    }])

    # predict
    proba      = model.predict_proba(input_df)[0][1]
    prediction = 'Churn' if proba >= 0.5 else 'No Churn'
    risk       = 'High' if proba >= 0.7 else 'Medium' if proba >= 0.4 else 'Low'
    risk_color = '#ff4b4b' if risk == 'High' else '#ffa500' if risk == 'Medium' else '#00c853'

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Prediction Result</h2>", unsafe_allow_html=True)

    if prediction == 'Churn':
        st.error(f'⚠️ This customer is likely to **CHURN**')
    else:
        st.success(f'✅ This customer is likely to **STAY**')

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric('Prediction',       prediction)
    with col_b:
        st.metric('Churn Probability', f'{proba:.1%}')
    with col_c:
        st.metric('Risk Level',        risk)

    # probability gauge
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('#### Churn Risk Gauge')
    st.progress(float(proba))

    # input summary
    with st.expander('📊 View Input Summary'):
        st.dataframe(input_df)
