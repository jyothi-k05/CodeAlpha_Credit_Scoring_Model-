# 💳 Credit Card Default Prediction

This project analyzes and predicts whether a customer is likely to default on their credit card payment using machine learning algorithms. The dataset used is the UCI Credit Card dataset.

---

## 📁 Project Structure

- `train_credit_model.py` → Python script to train and evaluate models (Logistic Regression, Decision Tree, Random Forest).
- `credit_score_app.py` → Streamlit web application for real-time creditworthiness prediction.
- `UCI_Credit_Card.csv` → Dataset used for training and testing.
- `requirements.txt` → List of required Python packages.

---

## 📊 Features

- Trains and evaluates 3 ML models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest
- Uses ROC AUC, Classification Report, and Confusion Matrix for evaluation.
- Streamlit app allows users to:
  - Enter customer financial data
  - Get prediction + risk score
  - Styled UI with background

---

## 🛠 How to Run

### 1️⃣ Train Models (optional)

```bash
python train_credit_model.py
