# 📡 Telco Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company 
using Logistic Regression. The model identifies customers likely to cancel 
their subscription so the business can intervene with retention offers in time.

## 🔗 Live App
[Telco Churn Predictor](still under deployment)  


---

## 📊 Results

| Model | CV AUC | Test AUC | Churn Recall | Features |
|---|---|---|---|---|
| LR Full | 0.8492 | 0.8474 | 80% | 56 |
| LR Reduced | 0.8499 | 0.8471 | 80% | 20 |

---

## 🔍 Key Findings

- **Contract type** is the strongest predictor — month-to-month customers churn 3x more
- **New customers** (0-12 months tenure) are the highest risk group
- **Fiber optic** customers churn more despite paying premium prices
- **Electronic check** payment method is associated with higher churn
- **Logistic Regression outperformed** Random Forest and Gradient Boosting
- **Feature selection** reduced 56 features to 20 with zero performance loss

---

## 🛠️ Tech Stack

- **Python** 3.11
- **Scikit-learn** — Logistic Regression, RFE, Cross Validation
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualisation
- **Streamlit** — web app deployment

---

## 📂 Dataset

- **Source**: IBM Sample Dataset — Telco Customer Churn
- **Size**: 7,043 customers, 19 features
- **Target**: Churn (Yes/No) — 26.5% churn rate
- **Features**: Contract type, tenure, monthly charges, internet service, payment method, and various add-on services

---

## 🚀 How to Run Locally

1. Clone the repo
```bash
git clone https://github.com/Aaryan-Tarambale/Telco-churn-prediction.git
cd Telco-churn-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app/app.py
```

---

## 📋 Approach

1. **Exploratory Data Analysis** — understand what drives churn
2. **Feature Engineering** — created 4 new features from business logic
3. **Preprocessing Pipeline** — StandardScaler + OneHotEncoder
4. **Baseline Model** — Logistic Regression with class balancing
5. **Feature Selection** — RFE reduced 56 → 20 features
6. **Validation** — 5-Fold Stratified Cross Validation
7. **Deployment** — Streamlit web app

---

## 💡 Business Recommendation

Focus retention efforts on customers who are:
- On a **month-to-month contract**
- In their **first 12 months** of tenure
- Using **Fiber optic** internet
- Paying via **electronic check**

These customers represent the highest churn risk and are the best 
candidates for targeted retention offers.

---

## 👤 Author
**Aaryan Tarambale**  
[GitHub](https://github.com/Aaryan-Tarambale)
