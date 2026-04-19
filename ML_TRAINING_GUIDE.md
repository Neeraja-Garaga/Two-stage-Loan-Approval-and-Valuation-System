# 🤖 Loan Approval ML Model Training Guide

## 📋 Overview

This guide will help you train an accurate machine learning model for loan approval predictions using real algorithms instead of rule-based scoring.

## 🎯 What This Achieves

- **Trained ML Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Hyperparameter Tuning**: Optimizes the best model automatically
- **Feature Engineering**: Creates advanced features from raw data
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Production-Ready**: Saves trained model for use in your Streamlit app

---

## 📦 Step 1: Install Dependencies

```bash
pip install -r requirements_ml.txt
```

This installs:
- scikit-learn (ML algorithms)
- pandas (data manipulation)
- matplotlib & seaborn (visualizations)
- joblib (model saving)

---

## 🎲 Step 2: Generate Training Data

Since your database likely has limited labeled data, generate synthetic samples:

```bash
python generate_synthetic_data.py
```

**What this does:**
- Creates 500 realistic loan applications
- Assigns Approved/Rejected labels based on sophisticated criteria
- Includes correlations (e.g., higher credit score → higher income)
- Inserts data into your `loan_system.db`

**Output:**
```
📊 GENERATING SYNTHETIC LOAN APPLICATION DATA
Generating 500 samples...
  Generated 50/500 samples...
  Generated 100/500 samples...
  ...
✅ Synthetic data generation complete!
```

---

## 🚀 Step 3: Train the Model

```bash
python train_model.py
```

**What this does:**

### 1. Data Loading
- Loads all Approved/Rejected applications from database
- Requires minimum 50 samples

### 2. Feature Engineering
- Calculates age from date of birth
- Creates financial ratios (loan-to-income, debt-to-income)
- Adds total assets and asset-to-loan ratio
- Creates credit score categories
- Encodes categorical variables

### 3. Model Training
Trains 4 different models:
- **Logistic Regression**: Simple, interpretable baseline
- **Random Forest**: Ensemble of decision trees (usually best)
- **Gradient Boosting**: Advanced ensemble method
- **SVM**: Support Vector Machine with RBF kernel

### 4. Model Comparison
- Evaluates each model on test data
- Compares accuracy, precision, recall, F1-score
- Performs 5-fold cross-validation
- Selects the best performer

### 5. Hyperparameter Optimization
- Runs GridSearchCV on the best model
- Tests different parameter combinations
- Finds optimal settings automatically

### 6. Evaluation & Visualization
Creates 3 visualizations in `models/` folder:
- **confusion_matrix.png**: Shows true vs predicted labels
- **feature_importance.png**: Most important factors for decisions
- **model_comparison.png**: Performance comparison charts

**Expected Output:**
```
🚀 LOAN APPROVAL ML MODEL TRAINING
==================================================================
📂 Loading data from database...
✅ Loaded 500 records
   - Approved: 325
   - Rejected: 175

🔧 Engineering features...
✅ Feature engineering complete

📊 Preparing data for training...
✅ Dataset prepared: 500 samples, 25 features

🤖 Training multiple models...

Training Logistic Regression...
  ✓ Accuracy: 0.8700
  ✓ Precision: 0.8889
  ✓ Recall: 0.9231
  ✓ F1-Score: 0.9057
  ✓ CV Score: 0.8650 (±0.0234)

Training Random Forest...
  ✓ Accuracy: 0.9300
  ✓ Precision: 0.9474
  ✓ Recall: 0.9487
  ✓ F1-Score: 0.9481
  ✓ CV Score: 0.9225 (±0.0189)

Training Gradient Boosting...
  ✓ Accuracy: 0.9200
  ✓ Precision: 0.9333
  ✓ Recall: 0.9487
  ✓ F1-Score: 0.9410
  ✓ CV Score: 0.9150 (±0.0201)

Training SVM...
  ✓ Accuracy: 0.8900
  ✓ Precision: 0.9091
  ✓ Recall: 0.9231
  ✓ F1-Score: 0.9160
  ✓ CV Score: 0.8800 (±0.0267)

🏆 Best Model: Random Forest
   F1-Score: 0.9481

⚙️ Optimizing Random Forest model...
Best parameters: {'max_depth': 20, 'max_features': 'sqrt', ...}
Best CV F1-Score: 0.9525

✅ Training complete! Model saved to: models/loan_approval_model.pkl
```

---

## 📊 Understanding the Results

### Key Metrics Explained:

**Accuracy**: % of correct predictions overall
- Good baseline, but can be misleading with imbalanced data

**Precision**: Of all loans we approved, what % were actually good?
- High precision = fewer bad loans approved
- Important for minimizing defaults

**Recall**: Of all good loans, what % did we approve?
- High recall = fewer good applicants rejected
- Important for customer satisfaction

**F1-Score**: Balanced metric (harmonic mean of precision & recall)
- Best overall performance indicator
- **Target: > 0.90 is excellent**

**Cross-Validation Score**: Performance on different data splits
- Tests model generalization
- Low std deviation = consistent performance

### Feature Importance:

The model will tell you which factors matter most:

**Typical Top Features:**
1. Credit score (most important)
2. Loan-to-income ratio
3. Total assets
4. Income
5. Existing loans
6. Age
7. Employment stability

---

## 🔄 Step 4: Integrate Model into Streamlit App

Replace the rule-based `predict_loan()` function with ML predictions:

```python
import joblib
import numpy as np

# Load trained model (do this once at startup)
model = joblib.load('models/loan_approval_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')
encoders = joblib.load('models/encoders.pkl')

def predict_loan_ml(data):
    """ML-based loan prediction"""
    
    # Calculate age
    from datetime import datetime
    dob = datetime.strptime(data['dob'], '%Y-%m-%d')
    age = (datetime.now() - dob).days // 365
    
    # Engineer features
    loan_to_income = data['loan_amount'] / (data['income'] + 1)
    debt_to_income = data['existing_loans'] / (data['income'] + 1)
    total_assets = (data['residential_assets'] + data['commercial_assets'] + 
                   data['luxury_assets'] + data['bank_assets'])
    asset_to_loan = total_assets / (data['loan_amount'] + 1)
    monthly_payment = data['loan_amount'] / (data['loan_term'] + 1)
    payment_to_income = (monthly_payment * 12) / (data['income'] + 1)
    
    # Credit categories
    credit_excellent = 1 if data['credit_score'] >= 750 else 0
    credit_good = 1 if 650 <= data['credit_score'] < 750 else 0
    credit_fair = 1 if 550 <= data['credit_score'] < 650 else 0
    
    employment_stable = 1 if data['job_years'] >= 2 else 0
    
    # Encode categoricals
    gender_enc = encoders['gender'].transform([data['gender']])[0]
    education_enc = encoders['education'].transform([data['education']])[0]
    employment_enc = encoders['employment'].transform([data['employment_type']])[0]
    purpose_enc = encoders['purpose'].transform([data['loan_purpose']])[0]
    
    # Create feature vector (must match training order)
    features = np.array([[
        age, gender_enc, education_enc, employment_enc,
        data['income'], data['loan_amount'], data['loan_term'], 
        data['credit_score'],
        data['residential_assets'], data['commercial_assets'],
        data['luxury_assets'], data['bank_assets'],
        data['job_years'], data['existing_loans'], purpose_enc,
        loan_to_income, debt_to_income, total_assets, asset_to_loan,
        monthly_payment, payment_to_income,
        credit_excellent, credit_good, credit_fair, employment_stable
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': 'Approved' if prediction == 1 else 'Rejected',
        'probability': probability[1],  # Probability of approval
        'confidence': max(probability)  # Confidence in prediction
    }
```

---

## 🔍 Step 5: Test the Model

Create a test script:

```python
# test_model.py
import joblib

model = joblib.load('models/loan_approval_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Test case: Good applicant
test_data = {
    'age': 35, 'gender_encoded': 0, 'education_encoded': 2,
    'employment_encoded': 0, 'income': 80000, 'loan_amount': 50000,
    'loan_term': 60, 'credit_score': 750, ...
}

prediction = model.predict(scaler.transform([list(test_data.values())]))
print(f"Prediction: {'Approved' if prediction[0] == 1 else 'Rejected'}")
```

---

## 📈 Step 6: Monitor and Improve

### Continuous Improvement:

1. **Collect Real Data**: As users apply, collect real outcomes
2. **Retrain Periodically**: Run `train_model.py` monthly with new data
3. **Track Performance**: Monitor approval/rejection rates
4. **A/B Testing**: Compare ML model vs rule-based system

### When to Retrain:

- Every 100 new labeled samples
- Monthly (if you have active users)
- When performance metrics drop
- When business rules change

---

## 🐛 Troubleshooting

### "Not enough training data"
- Run `generate_synthetic_data.py` first
- Need minimum 50 samples (Approved + Rejected)

### "Model accuracy is low"
- Check if data is realistic (synthetic vs real)
- Try generating more samples (increase NUM_SAMPLES)
- Check for data quality issues

### "ImportError: No module named 'sklearn'"
- Run: `pip install -r requirements_ml.txt`

### "Database file not found"
- Ensure `loan_system.db` exists in the same directory
- Run your Streamlit app first to create the database

---

## 🎯 Expected Performance

With good training data, you should achieve:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Accuracy | > 80% | > 85% | > 90% |
| Precision | > 80% | > 85% | > 90% |
| Recall | > 75% | > 80% | > 90% |
| F1-Score | > 80% | > 85% | > 92% |

**Random Forest typically achieves 90-95% accuracy on this task.**

---

## 📚 Files Created

After training, you'll have:

```
models/
├── loan_approval_model.pkl      # Trained model
├── scaler.pkl                    # Feature scaler
├── feature_names.pkl             # Feature list
├── encoders.pkl                  # Categorical encoders
├── confusion_matrix_*.png        # Confusion matrix
├── feature_importance.png        # Feature importance chart
└── model_comparison.png          # Model comparison
```

---

## 🚀 Production Checklist

Before deploying the ML model:

- [ ] Trained on at least 100 real samples
- [ ] F1-score > 0.85
- [ ] Tested on diverse applicant profiles
- [ ] Feature importance makes business sense
- [ ] Model files backed up
- [ ] Monitoring dashboard ready
- [ ] Fallback to rule-based system if model fails

---

## 💡 Next Steps

1. **Generate synthetic data**: `python generate_synthetic_data.py`
2. **Train the model**: `python train_model.py`
3. **Review visualizations**: Check `models/` folder
4. **Integrate into app**: Update `predict_loan()` function
5. **Test thoroughly**: Try various applicant profiles
6. **Deploy**: Replace rule-based system with ML model
7. **Monitor**: Track real-world performance

---

## 📞 Need Help?

Common issues and solutions are in the Troubleshooting section above. The model training script provides detailed output at each step to help diagnose any problems.

**Happy Training! 🎉**
