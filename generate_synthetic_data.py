"""
Generate Synthetic Loan Application Data for ML Training
Creates realistic loan application samples with Approved/Rejected labels
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Configuration
NUM_SAMPLES = 500  # Number of synthetic samples to generate
APPROVAL_RATE = 0.65  # Approximately 65% approval rate

# Realistic ranges for loan application data
CREDIT_SCORE_RANGE = (300, 850)
INCOME_RANGE = (20000, 200000)
LOAN_AMOUNT_RANGE = (5000, 500000)
LOAN_TERM_RANGE = (6, 360)  # months
AGE_RANGE = (21, 70)
JOB_YEARS_RANGE = (0, 40)

GENDERS = ['Male', 'Female', 'Other']
EDUCATIONS = ["High School", "Bachelor's", "Master's", "PhD", "Other"]
EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business", "Retired"]
LOAN_PURPOSES = ["Business", "Debt Consolidation", "Education", 
                 "Home Improvement", "Home Purchase", "Medical", "Personal", "Vehicle"]

def generate_correlated_data():
    """Generate realistic, correlated loan application data"""
    
    # Base applicant profile
    credit_score = np.random.normal(650, 80)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Income correlates with credit score and education
    education = random.choice(EDUCATIONS)
    education_multiplier = {
        "High School": 0.7,
        "Bachelor's": 1.0,
        "Master's": 1.3,
        "PhD": 1.5,
        "Other": 0.8
    }[education]
    
    base_income = np.random.normal(60000, 25000) * education_multiplier
    income_variation = (credit_score - 550) / 300  # Credit score affects income
    income = base_income * (1 + income_variation * 0.5)
    income = np.clip(income, 20000, 250000)
    
    # Loan amount should be reasonable relative to income
    max_reasonable_loan = income * np.random.uniform(3, 8)
    loan_amount = np.random.uniform(5000, min(max_reasonable_loan, 500000))
    
    # Age
    age = np.random.randint(21, 71)
    dob = datetime.now() - timedelta(days=age*365)
    
    # Job years should be less than working age
    max_job_years = min(age - 18, 40)
    job_years = np.random.randint(0, max(1, max_job_years))
    
    # Employment type
    if age > 60:
        employment_type = random.choices(
            EMPLOYMENT_TYPES, 
            weights=[0.2, 0.1, 0.1, 0.6]  # More likely retired
        )[0]
    else:
        employment_type = random.choices(
            EMPLOYMENT_TYPES,
            weights=[0.5, 0.3, 0.15, 0.05]  # Mostly salaried
        )[0]
    
    # Assets correlate with income and age
    asset_multiplier = (income / 50000) * (age / 40)
    residential_assets = np.random.exponential(100000) * asset_multiplier if random.random() > 0.3 else 0
    commercial_assets = np.random.exponential(50000) * asset_multiplier if random.random() > 0.7 else 0
    luxury_assets = np.random.exponential(20000) * (income / 60000) if random.random() > 0.5 else 0
    bank_assets = np.random.exponential(30000) * (credit_score / 650) if random.random() > 0.4 else 0
    
    # Existing loans inversely correlate with credit score
    existing_loan_probability = 1 - (credit_score - 300) / 550
    existing_loans = np.random.exponential(income * 0.5) * existing_loan_probability if random.random() > 0.5 else 0
    
    # Loan term
    loan_term = random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360])
    
    # Calculate approval probability based on realistic criteria
    approval_score = 0
    
    # Credit score weight (40 points max)
    if credit_score >= 750:
        approval_score += 40
    elif credit_score >= 650:
        approval_score += 25
    elif credit_score >= 550:
        approval_score += 10
    else:
        approval_score += 0
    
    # Income weight (30 points max)
    if income >= 100000:
        approval_score += 30
    elif income >= 50000:
        approval_score += 20
    elif income >= 30000:
        approval_score += 10
    else:
        approval_score += 0
    
    # Loan to income ratio (15 points max)
    loan_to_income = loan_amount / (income + 1)
    if loan_to_income < 2:
        approval_score += 15
    elif loan_to_income < 4:
        approval_score += 8
    elif loan_to_income < 6:
        approval_score += 3
    
    # Assets (15 points max)
    total_assets = residential_assets + commercial_assets + luxury_assets + bank_assets
    if total_assets >= loan_amount:
        approval_score += 15
    elif total_assets >= loan_amount * 0.5:
        approval_score += 8
    elif total_assets >= loan_amount * 0.25:
        approval_score += 4
    
    # Add some randomness
    approval_score += np.random.normal(0, 5)
    
    # Determine status
    # Threshold at 50, but add some noise to make it realistic
    noise = np.random.normal(0, 10)
    status = "Approved" if (approval_score + noise) >= 50 else "Rejected"
    
    return {
        'full_name': f"Person_{random.randint(1000, 9999)}",
        'dob': dob.strftime('%Y-%m-%d'),
        'gender': random.choice(GENDERS),
        'email': f"user{random.randint(1000, 9999)}@example.com",
        'phone': f"+1{random.randint(1000000000, 9999999999)}",
        'address': f"{random.randint(1, 999)} Main St",
        'education': education,
        'employment_type': employment_type,
        'income': round(income, 2),
        'loan_amount': round(loan_amount, 2),
        'loan_term': loan_term,
        'residential_assets': round(residential_assets, 2),
        'commercial_assets': round(commercial_assets, 2),
        'luxury_assets': round(luxury_assets, 2),
        'bank_assets': round(bank_assets, 2),
        'job_years': job_years,
        'existing_loans': round(existing_loans, 2),
        'credit_score': int(credit_score),
        'loan_purpose': random.choice(LOAN_PURPOSES),
        'status': status,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def insert_into_database(data_list):
    """Insert synthetic data into the database"""
    conn = sqlite3.connect('loan_system.db')
    cursor = conn.cursor()
    
    for data in data_list:
        try:
            cursor.execute("""
            INSERT INTO applications (
                user_id, full_name, dob, gender, email, phone, address,
                education, employment_type, income, loan_amount, loan_term,
                residential_assets, commercial_assets, luxury_assets, bank_assets,
                job_years, existing_loans, credit_score, loan_purpose, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                1,  # Default user_id for synthetic data
                data['full_name'], data['dob'], data['gender'], data['email'],
                data['phone'], data['address'], data['education'], data['employment_type'],
                data['income'], data['loan_amount'], data['loan_term'],
                data['residential_assets'], data['commercial_assets'],
                data['luxury_assets'], data['bank_assets'], data['job_years'],
                data['existing_loans'], data['credit_score'], data['loan_purpose'],
                data['status'], data['created_at']
            ))
        except Exception as e:
            print(f"Error inserting record: {e}")
            continue
    
    conn.commit()
    conn.close()

def main():
    print("="*70)
    print("📊 GENERATING SYNTHETIC LOAN APPLICATION DATA")
    print("="*70)
    print(f"\nGenerating {NUM_SAMPLES} samples...")
    
    # Generate data
    synthetic_data = []
    for i in range(NUM_SAMPLES):
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{NUM_SAMPLES} samples...")
        synthetic_data.append(generate_correlated_data())
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(synthetic_data)
    
    # Display statistics
    print("\n📈 Data Statistics:")
    print(f"   Total Samples: {len(df)}")
    print(f"   Approved: {len(df[df['status']=='Approved'])} ({len(df[df['status']=='Approved'])/len(df)*100:.1f}%)")
    print(f"   Rejected: {len(df[df['status']=='Rejected'])} ({len(df[df['status']=='Rejected'])/len(df)*100:.1f}%)")
    print(f"\n   Average Credit Score: {df['credit_score'].mean():.0f}")
    print(f"   Average Income: ${df['income'].mean():,.0f}")
    print(f"   Average Loan Amount: ${df['loan_amount'].mean():,.0f}")
    
    # Insert into database
    print("\n💾 Inserting data into database...")
    insert_into_database(synthetic_data)
    
    print("\n✅ Synthetic data generation complete!")
    print("\n💡 Next Steps:")
    print("   1. Run: python train_model.py")
    print("   2. The trained model will be saved to the models/ directory")
    print("   3. Update your Streamlit app to use the trained model")

if __name__ == "__main__":
    main()
