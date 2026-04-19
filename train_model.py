"""
Loan Approval ML Model Training Script
Train and evaluate multiple ML models for loan approval prediction
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
DATABASE_PATH = 'loan_system.db'
MODEL_SAVE_PATH = 'models/'
RANDOM_STATE = 42

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data_from_database():
    """Load loan application data from SQLite database"""
    print("📂 Loading data from database...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    query = """
    SELECT 
        full_name, dob, gender, education, employment_type,
        income, loan_amount, loan_term, residential_assets,
        commercial_assets, luxury_assets, bank_assets,
        job_years, existing_loans, credit_score, loan_purpose,
        status
    FROM applications
    WHERE status IN ('Approved', 'Rejected')
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"✅ Loaded {len(df)} records")
    print(f"   - Approved: {len(df[df['status']=='Approved'])}")
    print(f"   - Rejected: {len(df[df['status']=='Rejected'])}")
    
    return df

def engineer_features(df):
    """Create additional features from raw data"""
    print("\n🔧 Engineering features...")
    
    # Age from DOB
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (datetime.now() - df['dob']).dt.days // 365
    
    # Financial ratios
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
    df['debt_to_income_ratio'] = df['existing_loans'] / (df['income'] + 1)
    
    # Total assets
    df['total_assets'] = (df['residential_assets'] + df['commercial_assets'] + 
                         df['luxury_assets'] + df['bank_assets'])
    df['asset_to_loan_ratio'] = df['total_assets'] / (df['loan_amount'] + 1)
    
    # Monthly payment estimate
    df['monthly_payment'] = df['loan_amount'] / (df['loan_term'] + 1)
    df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / (df['income'] + 1)
    
    # Credit score categories
    df['credit_excellent'] = (df['credit_score'] >= 750).astype(int)
    df['credit_good'] = ((df['credit_score'] >= 650) & (df['credit_score'] < 750)).astype(int)
    df['credit_fair'] = ((df['credit_score'] >= 550) & (df['credit_score'] < 650)).astype(int)
    
    # Employment stability
    df['employment_stable'] = (df['job_years'] >= 2).astype(int)
    
    print("✅ Feature engineering complete")
    return df

def prepare_data(df):
    """Prepare data for ML training"""
    print("\n📊 Preparing data for training...")
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_education = LabelEncoder()
    le_employment = LabelEncoder()
    le_purpose = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['education_encoded'] = le_education.fit_transform(df['education'])
    df['employment_encoded'] = le_employment.fit_transform(df['employment_type'])
    df['purpose_encoded'] = le_purpose.fit_transform(df['loan_purpose'])
    
    # Save encoders
    joblib.dump({
        'gender': le_gender,
        'education': le_education,
        'employment': le_employment,
        'purpose': le_purpose
    }, f'{MODEL_SAVE_PATH}encoders.pkl')
    
    # Select features for training
    feature_columns = [
        'age', 'gender_encoded', 'education_encoded', 'employment_encoded',
        'income', 'loan_amount', 'loan_term', 'credit_score',
        'residential_assets', 'commercial_assets', 'luxury_assets', 'bank_assets',
        'job_years', 'existing_loans', 'purpose_encoded',
        'loan_to_income_ratio', 'debt_to_income_ratio', 'total_assets',
        'asset_to_loan_ratio', 'monthly_payment', 'payment_to_income_ratio',
        'credit_excellent', 'credit_good', 'credit_fair', 'employment_stable'
    ]
    
    X = df[feature_columns]
    y = (df['status'] == 'Approved').astype(int)
    
    print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_columns

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple ML models"""
    print("\n🤖 Training multiple models...\n")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall: {recall:.4f}")
        print(f"  ✓ F1-Score: {f1:.4f}")
        print(f"  ✓ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print()
    
    return results

def optimize_best_model(X_train, y_train):
    """Hyperparameter tuning for Random Forest (typically best for this task)"""
    print("\n⚙️  Optimizing Random Forest model...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{MODEL_SAVE_PATH}confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig(f'{MODEL_SAVE_PATH}feature_importance.png')
        plt.close()
        
        print("\n📊 Top 10 Most Important Features:")
        for idx in indices[-10:][::-1]:
            print(f"   {feature_names[idx]}: {importances[idx]:.4f}")

def plot_model_comparison(results):
    """Compare models visually"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        scores = [results[model][metric] for model in results]
        axes[idx].bar(results.keys(), scores, color='steelblue')
        axes[idx].set_title(f'{metric.capitalize()} Comparison')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_SAVE_PATH}model_comparison.png')
    plt.close()

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("🚀 LOAN APPROVAL ML MODEL TRAINING")
    print("="*70)
    
    # Create models directory
    import os
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Load data
    df = load_data_from_database()
    
    if len(df) < 50:
        print("\n⚠️  WARNING: Not enough training data!")
        print("   Need at least 50 labeled samples (Approved/Rejected)")
        print("   Current count:", len(df))
        print("\n💡 Recommendation: Generate synthetic data or collect more samples")
        return
    
    # Feature engineering
    df = engineer_features(df)
    
    # Prepare data
    X, y, feature_names = prepare_data(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, f'{MODEL_SAVE_PATH}scaler.pkl')
    
    # Train multiple models
    results = train_multiple_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
    
    # Optimize the best model if it's Random Forest
    if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name:
        optimized_model = optimize_best_model(X_train_scaled, y_train)
        y_pred_opt = optimized_model.predict(X_test_scaled)
        opt_f1 = f1_score(y_test, y_pred_opt)
        
        if opt_f1 > results[best_model_name]['f1']:
            print(f"\n✅ Optimized model is better! F1: {opt_f1:.4f}")
            best_model = optimized_model
            best_model_name = f"{best_model_name} (Optimized)"
    
    # Save the best model
    joblib.dump(best_model, f'{MODEL_SAVE_PATH}loan_approval_model.pkl')
    joblib.dump(feature_names, f'{MODEL_SAVE_PATH}feature_names.pkl')
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_confusion_matrix(y_test, best_model.predict(X_test_scaled), best_model_name)
    plot_feature_importance(best_model, feature_names)
    plot_model_comparison(results)
    
    # Final evaluation report
    print("\n" + "="*70)
    print("📋 FINAL MODEL EVALUATION REPORT")
    print("="*70)
    print(f"\nModel: {best_model_name}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test_scaled), 
                                target_names=['Rejected', 'Approved']))
    
    print("\n✅ Training complete! Model saved to:", f'{MODEL_SAVE_PATH}loan_approval_model.pkl')
    print("✅ Visualizations saved to:", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
