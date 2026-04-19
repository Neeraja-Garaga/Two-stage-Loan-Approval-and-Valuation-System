import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# ==================== DATABASE SETUP ====================

def init_db():
    conn = sqlite3.connect('loan_system.db', check_same_thread=False)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  email TEXT UNIQUE,
                  password TEXT,
                  role TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS applications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  full_name TEXT,
                  dob TEXT,
                  gender TEXT,
                  email TEXT,
                  phone TEXT,
                  address TEXT,
                  education TEXT,
                  employment_type TEXT,
                  income REAL,
                  loan_amount REAL,
                  loan_term INTEGER,
                  residential_assets REAL,
                  commercial_assets REAL,
                  luxury_assets REAL,
                  bank_assets REAL,
                  job_years INTEGER,
                  existing_loans REAL,
                  credit_score INTEGER,
                  loan_purpose TEXT,
                  status TEXT,
                  created_at TEXT)''')
    
    conn.commit()
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ==================== ML MODEL ====================

def predict_loan(data):
    score = 0
    
    # Credit score weight
    if data['credit_score'] >= 750:
        score += 40
    elif data['credit_score'] >= 650:
        score += 25
    elif data['credit_score'] >= 550:
        score += 10
    
    # Income weight
    if data['income'] >= 100000:
        score += 30
    elif data['income'] >= 50000:
        score += 20
    elif data['income'] >= 30000:
        score += 10
    
    # Loan to income ratio
    loan_to_income = data['loan_amount'] / (data['income'] + 1)
    if loan_to_income < 2:
        score += 15
    elif loan_to_income < 4:
        score += 8
    
    # Assets
    total_assets = (data['residential_assets'] + data['commercial_assets'] + 
                   data['luxury_assets'] + data['bank_assets'])
    if total_assets >= data['loan_amount']:
        score += 15
    elif total_assets >= data['loan_amount'] * 0.5:
        score += 8
    
    return "Approved" if score >= 50 else "Rejected"

def calculate_suggested_loan_amount(data):
    """Calculate appropriate loan amount based on user's financial profile"""
    
    # Base calculation on income (conservative approach: 2x annual income)
    income_based = data['income'] * 2
    
    # Adjust based on credit score
    if data['credit_score'] >= 750:
        credit_multiplier = 1.5
    elif data['credit_score'] >= 650:
        credit_multiplier = 1.2
    elif data['credit_score'] >= 550:
        credit_multiplier = 1.0
    else:
        credit_multiplier = 0.7
    
    # Consider existing debt
    debt_adjusted = income_based - data['existing_loans']
    if debt_adjusted < 0:
        debt_adjusted = income_based * 0.3
    
    # Calculate with assets
    total_assets = (data['residential_assets'] + data['commercial_assets'] + 
                   data['luxury_assets'] + data['bank_assets'])
    
    # Suggested amount is lower of: adjusted income-based or 70% of assets
    suggested = min(debt_adjusted * credit_multiplier, total_assets * 0.7) if total_assets > 0 else debt_adjusted * credit_multiplier
    
    # Round to nearest thousand
    suggested = round(suggested / 1000) * 1000
    
    # Ensure minimum of $5,000
    return max(5000, suggested)

def get_improvement_suggestions(data):
    suggestions = []
    
    if data['credit_score'] < 750:
        suggestions.append(f"📈 Improve credit score from {data['credit_score']} to 750+ for better approval chances")
    
    if data['income'] < 100000:
        suggestions.append(f"💰 Current income: ${data['income']:,.0f}. Higher income increases approval probability")
    
    loan_to_income = data['loan_amount'] / (data['income'] + 1)
    if loan_to_income > 2:
        suggestions.append(f"⚖️ Loan-to-income ratio is {loan_to_income:.1f}. Try to keep it below 2.0")
    
    total_assets = (data['residential_assets'] + data['commercial_assets'] + 
                   data['luxury_assets'] + data['bank_assets'])
    if total_assets < data['loan_amount'] * 0.5:
        suggestions.append(f"🏦 Total assets: ${total_assets:,.0f}. Consider increasing to at least 50% of loan amount")
    
    if data['existing_loans'] > 0:
        suggestions.append(f"💳 Existing loans: ${data['existing_loans']:,.0f}. Lower debt improves approval odds")
    
    return suggestions if suggestions else ["✅ Your profile looks strong! Keep maintaining good financial habits."]

# ==================== CSS STYLING ====================

def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .loan-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .loan-card h3 {
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .loan-card p {
        color: #34495E;
        margin: 0.3rem 0;
    }
    .header-title {
        color: #2C3E50;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
    }
    .subtitle {
        color: #7F8C8D;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-approved {
        color: #27AE60;
        font-weight: 600;
    }
    .status-rejected {
        color: #E74C3C;
        font-weight: 600;
    }
    .status-pending {
        color: #F39C12;
        font-weight: 600;
    }
    /* Fix for input labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label, 
    .stDateInput label, .stTextArea label {
        color: #2C3E50 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Table styling - white background */
    .stDataFrame {
        background-color: white !important;
    }
    
    div[data-testid="stDataFrame"] {
        background-color: white !important;
    }
    
    div[data-testid="stDataFrame"] > div {
        background-color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== AUTHENTICATION ====================

def register_user(conn, username, email, password, role):
    try:
        c = conn.cursor()
        hashed_pw = hash_password(password)
        c.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                 (username, email, hashed_pw, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(conn, email, password, role):
    c = conn.cursor()
    hashed_pw = hash_password(password)
    c.execute("SELECT * FROM users WHERE email=? AND password=? AND role=?",
             (email, hashed_pw, role))
    return c.fetchone()

# ==================== HOME PAGE ====================

def show_home_page():
    st.markdown('<p class="header-title">🏦 Loan Approval & Valuation System</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📝 Register"):
            st.session_state.page = "register"
            st.rerun()
    with col2:
        if st.button("🔐 Login"):
            st.session_state.page = "login"
            st.rerun()
    with col3:
        if st.button("ℹ️ About Us"):
            st.info("Professional loan approval system powered by intelligent analytics.")
    with col4:
        if st.button("📧 Contact Us"):
            st.info("Email: support@loanapproval.com | Phone: 1-800-LOAN-APP")
    
    st.markdown("---")
    
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: white; border-radius: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.1);'>
            <div style='font-size: 5rem;'>🏦</div>
            <h1 style='color: #2C3E50; margin: 1rem 0;'>From Application to Approval</h1>
            <h3 style='color: #7F8C8D;'>Powered by Intelligent Analytics</h3>
            <p style='color: #95A5A6; margin-top: 2rem;'>Loan Approval and Valuation System</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>⚡</div>
            <h3>Fast Processing</h3>
            <p>Get decisions in minutes</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>🤖</div>
            <h3>AI-Powered</h3>
            <p>Smart loan evaluation</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>🔒</div>
            <h3>Secure</h3>
            <p>Your data is safe</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== REGISTER PAGE ====================

def show_register_page(conn):
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

    .stApp {
        background-color: #3d5a6b !important;
        background-image: none !important;
    }
    section[data-testid="stMain"] > div {
        background-color: #3d5a6b !important;
    }
    .block-container {
        background-color: #3d5a6b !important;
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 480px !important;
        margin: 0 auto !important;
    }
    header[data-testid="stHeader"] {
        background-color: #3d5a6b !important;
        box-shadow: none !important;
    }
    .stTextInput > label,
    .stSelectbox > label { display: none !important; }

    /* Arrow button inside the header bar */
    /* Bank icon - centered */
    .reg-icon-circle {
        width: 60px; height: 60px;
        background: #e8edf0;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 4px auto;
        font-size: 1.8rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.25);
    }
    .reg-welcome-text {
        text-align: center;
        color: #c8d8e0;
        font-size: 0.75rem;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        font-family: 'Nunito', sans-serif;
        margin-bottom: 8px;
    }
    /* Compact title box */
    .reg-title-box {
        background: #e8edf0;
        border-radius: 14px;
        padding: 10px 20px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.18);
    }
    .reg-title-box h2 {
        color: #3d5a6b;
        font-size: 1.05rem;
        font-weight: 700;
        font-family: 'Nunito', sans-serif;
        margin: 0;
    }

    /* Pill input fields */
    .stTextInput > div > div > input {
        background-color: #2e4555 !important;
        border: none !important;
        border-radius: 50px !important;
        color: #c8d8e0 !important;
        padding: 10px 20px !important;
        font-size: 0.9rem !important;
        font-family: 'Nunito', sans-serif !important;
        height: 46px !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.22) !important;
        caret-color: white !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #7a9aac !important;
        font-size: 0.88rem !important;
    }
    .stTextInput > div > div > input:focus {
        outline: none !important;
        border: 1.5px solid #7ab8d0 !important;
        box-shadow: 0 0 0 3px rgba(122,184,208,0.12) !important;
    }

    /* Selectbox pill */
    .stSelectbox > div > div {
        background-color: #2e4555 !important;
        border: none !important;
        border-radius: 50px !important;
        height: 46px !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.22) !important;
    }
    .stSelectbox > div > div > div[data-baseweb="select"] > div {
        background-color: #2e4555 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0 20px !important;
        color: #c8d8e0 !important;
        font-family: 'Nunito', sans-serif !important;
        font-size: 0.88rem !important;
    }
    .stSelectbox svg { fill: #7a9aac !important; }

    /* Primary button */
    .stButton > button {
        background-color: #1e3040 !important;
        color: #e8edf0 !important;
        border: none !important;
        border-radius: 50px !important;
        height: 46px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        font-family: 'Nunito', sans-serif !important;
        letter-spacing: 1px !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.32) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #16232f !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.42) !important;
        transform: translateY(-1px) !important;
    }

    .nav-link-text {
        text-align: center;
        color: #a0bfcc;
        font-size: 0.82rem;
        font-family: 'Nunito', sans-serif;
        margin: 4px 0;
    }
    .alert-success {
        background: rgba(56,161,105,0.2);
        border: 1px solid rgba(56,161,105,0.5);
        border-radius: 12px; padding: 8px 16px;
        color: #68d391; font-family: 'Nunito', sans-serif;
        font-size: 0.84rem; font-weight: 600; margin-top: 6px;
    }
    .alert-error {
        background: rgba(220,53,69,0.2);
        border: 1px solid rgba(220,53,69,0.45);
        border-radius: 12px; padding: 8px 16px;
        color: #fc8181; font-family: 'Nunito', sans-serif;
        font-size: 0.84rem; font-weight: 600; margin-top: 6px;
    }
    .alert-warning {
        background: rgba(236,201,75,0.15);
        border: 1px solid rgba(236,201,75,0.4);
        border-radius: 12px; padding: 8px 16px;
        color: #f6e05e; font-family: 'Nunito', sans-serif;
        font-size: 0.84rem; font-weight: 600; margin-top: 6px;
    }
    div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # Bank icon + welcome + compact title
    st.markdown("""
        <div style='text-align:center;'>
            <div class='reg-icon-circle'>🏦</div>
            <p class='reg-welcome-text'>Welcome to Loan Portal</p>
        </div>
        <div class='reg-title-box'><h2>Create Account</h2></div>
    """, unsafe_allow_html=True)

    # Fields
    full_name = st.text_input("fn", placeholder="Full Name",    label_visibility="collapsed", key="reg_fullname")
    email     = st.text_input("em", placeholder="Email ID",     label_visibility="collapsed", key="reg_email")
    password  = st.text_input("pw", placeholder="Password",     type="password", label_visibility="collapsed", key="reg_password")
    
    # Auto-assign role as Applicant (Admin roles are created by administration only)
    role = "Applicant"
    
    # Info message
    st.markdown("""
    <div style='background:rgba(74,144,226,0.15);padding:8px 16px;border-radius:10px;
                margin:6px 0;border-left:3px solid #4A90E2;'>
        <p style='color:#c8d8e0;margin:0;font-size:0.8rem;font-family:Nunito,sans-serif;'>
            ℹ️ All new registrations are created as <strong>Applicant</strong> accounts. 
            Admin access is granted by administration only.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if 'reg_error'   not in st.session_state: st.session_state.reg_error   = ""
    if 'reg_success' not in st.session_state: st.session_state.reg_success = ""

    if st.button("Register", use_container_width=True, key="reg_submit"):
        import re
        if not full_name or not email or not password:
            st.session_state.reg_error = "⚠️ All fields are required."
            st.session_state.reg_success = ""
        elif not re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', email):
            st.session_state.reg_error = "⚠️ Enter a valid email address."
            st.session_state.reg_success = ""
        elif len(password) < 6:
            st.session_state.reg_error = "⚠️ Password must be at least 6 characters."
            st.session_state.reg_success = ""
        else:
            if register_user(conn, full_name, email, password, role):
                st.session_state.reg_success = "✅ Account created! Please login."
                st.session_state.reg_error   = ""
                import time; time.sleep(0.5)
                st.session_state.page = "login"
                st.rerun()
            else:
                st.session_state.reg_error   = "❌ Email already registered."
                st.session_state.reg_success = ""

    if st.session_state.reg_error:
        st.markdown(f"<div class='alert-error'>{st.session_state.reg_error}</div>", unsafe_allow_html=True)
    if st.session_state.reg_success:
        st.markdown(f"<div class='alert-success'>{st.session_state.reg_success}</div>", unsafe_allow_html=True)

    st.markdown("<p class='nav-link-text'>Already have an account?</p>", unsafe_allow_html=True)

    if st.button("Login →", use_container_width=True, key="goto_login"):
        st.session_state.page = "login"
        st.rerun()

# ==================== LOGIN PAGE ====================

def show_login_page(conn):
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

    .stApp {
        background-color: #3d5a6b !important;
        background-image: none !important;
    }
    section[data-testid="stMain"] > div {
        background-color: #3d5a6b !important;
    }
    .block-container {
        background-color: #3d5a6b !important;
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 480px !important;
        margin: 0 auto !important;
    }
    header[data-testid="stHeader"] {
        background-color: #3d5a6b !important;
        box-shadow: none !important;
    }
    .stTextInput > label,
    .stSelectbox > label { display: none !important; }

    /* Arrow button inside the header bar */
    /* Bank icon */
    .login-icon-circle {
        width: 60px; height: 60px;
        background: #e8edf0;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 4px auto;
        font-size: 1.8rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.25);
    }
    .login-welcome-text {
        text-align: center;
        color: #c8d8e0;
        font-size: 0.75rem;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        font-family: 'Nunito', sans-serif;
        margin-bottom: 8px;
    }
    /* Compact title box */
    .login-title-box {
        background: #e8edf0;
        border-radius: 14px;
        padding: 10px 20px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.18);
    }
    .login-title-box h2 {
        color: #3d5a6b;
        font-size: 1.05rem;
        font-weight: 700;
        font-family: 'Nunito', sans-serif;
        margin: 0;
    }

    /* Pill inputs */
    .stTextInput > div > div > input {
        background-color: #2e4555 !important;
        border: none !important;
        border-radius: 50px !important;
        color: #c8d8e0 !important;
        padding: 10px 20px !important;
        font-size: 0.9rem !important;
        font-family: 'Nunito', sans-serif !important;
        height: 46px !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.22) !important;
        caret-color: white !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #7a9aac !important;
        font-size: 0.88rem !important;
    }
    .stTextInput > div > div > input:focus {
        outline: none !important;
        border: 1.5px solid #7ab8d0 !important;
        box-shadow: 0 0 0 3px rgba(122,184,208,0.12) !important;
    }

    /* Selectbox pill */
    .stSelectbox > div > div {
        background-color: #2e4555 !important;
        border: none !important;
        border-radius: 50px !important;
        height: 46px !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.22) !important;
    }
    .stSelectbox > div > div > div[data-baseweb="select"] > div {
        background-color: #2e4555 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0 20px !important;
        color: #c8d8e0 !important;
        font-family: 'Nunito', sans-serif !important;
        font-size: 0.88rem !important;
    }
    .stSelectbox svg { fill: #7a9aac !important; }

    /* Primary button */
    .stButton > button {
        background-color: #1e3040 !important;
        color: #e8edf0 !important;
        border: none !important;
        border-radius: 50px !important;
        height: 46px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        font-family: 'Nunito', sans-serif !important;
        letter-spacing: 1px !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.32) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #16232f !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.42) !important;
        transform: translateY(-1px) !important;
    }

    .nav-link-text {
        text-align: center;
        color: #a0bfcc;
        font-size: 0.82rem;
        font-family: 'Nunito', sans-serif;
        margin: 4px 0;
    }
    .alert-error {
        background: rgba(220,53,69,0.2);
        border: 1px solid rgba(220,53,69,0.45);
        border-radius: 12px; padding: 8px 16px;
        color: #fc8181; font-family: 'Nunito', sans-serif;
        font-size: 0.84rem; font-weight: 600; margin-top: 6px;
    }
    .alert-warning {
        background: rgba(236,201,75,0.15);
        border: 1px solid rgba(236,201,75,0.4);
        border-radius: 12px; padding: 8px 16px;
        color: #f6e05e; font-family: 'Nunito', sans-serif;
        font-size: 0.84rem; font-weight: 600; margin-top: 6px;
    }
    div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # Bank icon + welcome + compact title
    st.markdown("""
        <div style='text-align:center;'>
            <div class='login-icon-circle'>🏦</div>
            <p class='login-welcome-text'>Welcome to Loan Portal</p>
        </div>
        <div class='login-title-box'><h2>Login</h2></div>
    """, unsafe_allow_html=True)

    # Fields
    email    = st.text_input("em_l", placeholder="Email ID",  label_visibility="collapsed", key="login_email")
    password = st.text_input("pw_l", placeholder="Password",  type="password", label_visibility="collapsed", key="login_password")
    role     = st.selectbox("rl_l", ["Applicant", "Admin"],   label_visibility="collapsed", key="login_role")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if 'login_error' not in st.session_state:
        st.session_state.login_error = ""

    if st.button("Login", use_container_width=True, key="login_submit"):
        import re
        if not email or not password:
            st.session_state.login_error = "⚠️ All fields are required."
        elif not re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', email):
            st.session_state.login_error = "⚠️ Enter a valid email address."
        elif len(password) < 6:
            st.session_state.login_error = "⚠️ Password must be at least 6 characters."
        else:
            user = login_user(conn, email, password, role)
            if user:
                st.session_state.logged_in   = True
                st.session_state.user_id     = user[0]
                st.session_state.username    = user[1]
                st.session_state.email       = user[2]
                st.session_state.role        = user[4]
                st.session_state.page        = "dashboard"
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "❌ Invalid credentials. Please try again."

    if st.session_state.login_error:
        st.markdown(f"<div class='alert-error'>{st.session_state.login_error}</div>", unsafe_allow_html=True)

    st.markdown("<p class='nav-link-text'>Don't have an account?</p>", unsafe_allow_html=True)

    if st.button("Register →", use_container_width=True, key="goto_register"):
        st.session_state.page = "register"
        st.rerun()

# ==================== APPLICANT DASHBOARD ====================

def show_applicant_dashboard(conn):
    st.sidebar.title(f"👤 {st.session_state.username}")
    
    if st.sidebar.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    menu = st.sidebar.radio("Navigation", 
                           ["🏠 Dashboard", "📝 Application Form", "📋 My Loans", "📊 Analytics"])
    
    if menu == "🏠 Dashboard":
        show_applicant_home()
    elif menu == "📝 Application Form":
        show_application_form(conn)
    elif menu == "📋 My Loans":
        show_my_loans(conn)
    elif menu == "📊 Analytics":
        show_applicant_analytics(conn)

def show_applicant_home():
    st.markdown(f'<p class="header-title">Welcome Back, {st.session_state.username}! 👋</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Manage your loan applications from your dashboard</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>📝</div>
            <h3>Apply</h3>
            <p>Submit new application</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>📋</div>
            <h3>Track</h3>
            <p>View your loans</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>📊</div>
            <h3>Analyze</h3>
            <p>See insights</p>
        </div>
        """, unsafe_allow_html=True)

def show_application_form(conn):
    st.markdown('<p class="header-title">📝 New Loan Application</p>', unsafe_allow_html=True)
    
    # Add custom CSS for Google Form style
    st.markdown("""
    <style>
    .form-container {
        max-width: 800px;
        margin: 0 auto;
        background: white;
        padding: 3rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .form-title {
        color: #1a73e8;
        font-size: 2rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 1rem;
    }
    .section-title {
        color: #202124;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stForm"] {
        border: none !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input {
        border: none !important;
        border-bottom: 1px solid #dadce0 !important;
        border-radius: 0 !important;
        padding: 8px 0 !important;
        background: transparent !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus,
    .stDateInput > div > div > input:focus {
        border-bottom: 2px solid #1a73e8 !important;
        box-shadow: none !important;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label, 
    .stDateInput label, .stTextArea label {
        color: #202124 !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered container
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div style='width: 60px; height: 60px; background: #1a73e8; border-radius: 50%; 
                        display: inline-flex; align-items: center; justify-content: center; margin-bottom: 1rem;'>
                <span style='color: white; font-size: 2rem;'>🏦</span>
            </div>
            <h1 style='color: #1a73e8; margin: 0; font-weight: 400;'>Loan Application Form</h1>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("loan_application"):
            # Personal Information Section
            st.markdown("<p class='section-title'>Personal Information</p>", unsafe_allow_html=True)
            full_name = st.text_input("Full Name")
            dob = st.date_input("Date of Birth")
            gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Other"])
            email = st.text_input("Email")
            phone = st.text_input("Phone")
            address = st.text_area("Address", height=100)
            
            # Education & Employment Section
            st.markdown("<p class='section-title'>Education & Employment</p>", unsafe_allow_html=True)
            education = st.selectbox("Education*", 
                                    ["Select", "High School", "Bachelor's", "Master's", "PhD", "Other"])
            employment_type = st.selectbox("Employment Type", 
                                          ["Select", "Salaried", "Self-Employed", "Business", "Retired"])
            income = st.number_input("Income", min_value=0.0, step=1000.0, value=0.0)
            job_years = st.number_input("Job Years", min_value=0, max_value=50, value=0)
            
            # Loan Details Section
            st.markdown("<p class='section-title'>Loan Details</p>", unsafe_allow_html=True)
            loan_amount = st.number_input("Loan Amount", min_value=1000.0, step=1000.0, value=1000.0)
            loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=360, step=6, value=12)
            loan_purpose = st.selectbox("Loan Purpose", 
                                       ["Business", "Debt Consolidation", "Education", 
                                        "Home Improvement", "Home Purchase", "Medical", 
                                        "Personal", "Vehicle"])
            
            # Assets Section
            st.markdown("<p class='section-title'>Assets Information</p>", unsafe_allow_html=True)
            residential_assets = st.number_input("Residential Assets Value", min_value=0.0, step=10000.0, value=0.0)
            commercial_assets = st.number_input("Commercial Assets Value", min_value=0.0, step=10000.0, value=0.0)
            luxury_assets = st.number_input("Luxury Assets Value", min_value=0.0, step=5000.0, value=0.0)
            bank_assets = st.number_input("Bank Asset Value", min_value=0.0, step=5000.0, value=0.0)
            
            # Financial Information Section
            st.markdown("<p class='section-title'>Financial Information</p>", unsafe_allow_html=True)
            existing_loans = st.number_input("Existing Loans", min_value=0.0, step=1000.0, value=0.0)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=300)
            
            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                submitted = st.form_submit_button("Submit Application", use_container_width=True, type="primary")
            
        if submitted:
            if full_name and email and phone and gender != "Select..." and education != "Select" and employment_type != "Select":
                c = conn.cursor()
                c.execute("""INSERT INTO applications 
                           (user_id, full_name, dob, gender, email, phone, address, education, 
                            employment_type, income, loan_amount, loan_term, residential_assets,
                            commercial_assets, luxury_assets, bank_assets, job_years, 
                            existing_loans, credit_score, loan_purpose, status, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (st.session_state.user_id, full_name, str(dob), gender, email, phone,
                          address, education, employment_type, income, loan_amount, loan_term,
                          residential_assets, commercial_assets, luxury_assets, bank_assets,
                          job_years, existing_loans, credit_score, loan_purpose, "Pending",
                          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                
                # Display success message with visible text
                st.markdown("""
                <div style='background-color: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin-top: 1rem;'>
                    <p style='color: #155724; margin: 0; font-weight: 600;'>✅ Application submitted successfully! Your application is now pending review.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style='background-color: #d1ecf1; padding: 1rem; border-radius: 8px; border-left: 4px solid #0c5460; margin-top: 0.5rem;'>
                    <p style='color: #0c5460; margin: 0;'>💡 You can track your application status in the 'My Loans' section.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
            else:
                st.markdown("""
                <div style='background-color: #f8d7da; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545; margin-top: 1rem;'>
                    <p style='color: #721c24; margin: 0; font-weight: 600;'>❌ Please fill all required fields</p>
                </div>
                """, unsafe_allow_html=True)

def show_my_loans(conn):
    st.markdown('<p class="header-title">📋 My Loan Applications</p>', unsafe_allow_html=True)
    
    c = conn.cursor()
    c.execute("SELECT * FROM applications WHERE user_id=? ORDER BY created_at DESC", (st.session_state.user_id,))
    loans = c.fetchall()
    
    if loans:
        for loan in loans:
            status_class = f"status-{loan[21].lower()}"
            st.markdown(f"""
            <div class='loan-card'>
                <h3>💰 Loan Amount: ${loan[11]:,.0f}</h3>
                <p><strong>Applicant Name:</strong> {loan[2]}</p>
                <p><strong>Status:</strong> <span class='{status_class}'>{loan[21]}</span></p>
                <p><strong>Applied On:</strong> {loan[22]}</p>
                <p><strong>Loan Term:</strong> {loan[12]} months</p>
                <p><strong>Purpose:</strong> {loan[20]}</p>
                <p><strong>Credit Score:</strong> {loan[19]}</p>
                <p><strong>Annual Income:</strong> ${loan[10]:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"📊 View Analytics for Loan #{loan[0]}", key=f"view_{loan[0]}"):
                st.session_state.selected_loan = loan[0]
                show_loan_analytics(conn, loan)
            
            st.markdown("---")
    else:
        st.info("📭 No applications found. Submit your first application using the Application Form!")

def show_loan_analytics(conn, loan):
    st.markdown("---")
    st.markdown("### 📊 Detailed Loan Analytics")
    
    # Prepare data
    data = {
        'credit_score': loan[19],
        'income': loan[10],
        'loan_amount': loan[11],
        'residential_assets': loan[13],
        'commercial_assets': loan[14],
        'luxury_assets': loan[15],
        'bank_assets': loan[16],
        'existing_loans': loan[18]
    }
    
    # Pie chart for assets distribution
    col1, col2 = st.columns(2)
    
    with col1:
        asset_labels = ['Residential', 'Commercial', 'Luxury', 'Bank']
        asset_values = [loan[13], loan[14], loan[15], loan[16]]
        
        fig_pie = go.Figure(data=[go.Pie(labels=asset_labels, values=asset_values, hole=0.3)])
        fig_pie.update_layout(title="Asset Distribution", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart comparison
        categories = ['Credit Score', 'Income/1000', 'Loan Amount/1000']
        user_values = [loan[19], loan[10]/1000, loan[11]/1000]
        ideal_values = [750, 100, 50]
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Your Values', x=categories, y=user_values, marker_color='#4A90E2'),
            go.Bar(name='Ideal Values', x=categories, y=ideal_values, marker_color='#27AE60')
        ])
        fig_bar.update_layout(title="Profile vs Ideal", barmode='group', height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ML Prediction
    prediction = predict_loan(data)
    suggestions = get_improvement_suggestions(data)
    
    st.markdown("### 🤖 AI Analysis")
    
    if prediction == "Approved":
        st.success(f"✅ Prediction: **{prediction}**")
    else:
        st.error(f"❌ Prediction: **{prediction}**")
    
    st.markdown("### 💡 Improvement Suggestions")
    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")
    
    st.markdown("### 🔍 Transparency")
    st.info("""
    **How the model works:**
    - Credit score (40 points max)
    - Income level (30 points max)
    - Loan-to-income ratio (15 points max)
    - Asset coverage (15 points max)
    - **Threshold:** 50+ points for approval
    """)

def show_applicant_analytics(conn):
    st.markdown('<p class="header-title">📊 My Analytics Dashboard</p>', unsafe_allow_html=True)
    
    c = conn.cursor()
    c.execute("SELECT * FROM applications WHERE user_id=?", (st.session_state.user_id,))
    loans = c.fetchall()
    
    if not loans:
        st.info("📭 No data available. Submit applications to see analytics.")
        return
    
    # Calculate metrics
    total_apps = len(loans)
    approved = len([l for l in loans if l[21] == "Approved"])
    rejected = len([l for l in loans if l[21] == "Rejected"])
    pending = len([l for l in loans if l[21] == "Pending"])
    avg_loan = sum([l[11] for l in loans]) / total_apps if total_apps > 0 else 0
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>📝</div>
            <h2 style='color: #4A90E2;'>{total_apps}</h2>
            <p style='color: #34495E;'>Total Applications</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>✅</div>
            <h2 style='color: #27AE60;'>{approved}</h2>
            <p style='color: #34495E;'>Approved</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>❌</div>
            <h2 style='color: #E74C3C;'>{rejected}</h2>
            <p style='color: #34495E;'>Rejected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>⏳</div>
            <h2 style='color: #F39C12;'>{pending}</h2>
            <p style='color: #34495E;'>Pending</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional metrics row
    col1, col2 = st.columns(2)
    
    with col1:
        approval_rate = (approved / total_apps * 100) if total_apps > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>📈</div>
            <h2 style='color: #27AE60;'>{approval_rate:.1f}%</h2>
            <p style='color: #34495E;'>Approval Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>💰</div>
            <h2 style='color: #9B59B6;'>${avg_loan:,.0f}</h2>
            <p style='color: #34495E;'>Average Loan Amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        status_df = pd.DataFrame({
            'Status': ['Approved', 'Rejected', 'Pending'],
            'Count': [approved, rejected, pending]
        })
        fig_status = px.pie(status_df, values='Count', names='Status', 
                           color='Status',
                           color_discrete_map={'Approved': '#27AE60', 
                                              'Rejected': '#E74C3C',
                                              'Pending': '#F39C12'},
                           title="Application Status Distribution")
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        # Loan amounts over time
        loan_df = pd.DataFrame([(l[22], l[11]) for l in loans], 
                              columns=['Date', 'Amount'])
        fig_timeline = px.line(loan_df, x='Date', y='Amount', 
                              title="Loan Amount Timeline",
                              markers=True)
        fig_timeline.update_traces(line_color='#4A90E2', marker=dict(size=10))
        st.plotly_chart(fig_timeline, use_container_width=True)

# ==================== ADMIN DASHBOARD ====================

def show_admin_dashboard(conn):
    st.sidebar.title(f"👨‍💼 Admin: {st.session_state.username}")
    
    if st.sidebar.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    menu = st.sidebar.radio("Navigation", 
                           ["🏠 Dashboard", "⏳ Pending Applications", "📋 All Applications", "📊 Analytics"])
    
    if menu == "🏠 Dashboard":
        show_admin_home(conn)
    elif menu == "⏳ Pending Applications":
        show_pending_applications(conn)
    elif menu == "📋 All Applications":
        show_all_applications(conn)
    elif menu == "📊 Analytics":
        show_admin_analytics(conn)

def show_admin_home(conn):
    st.markdown(f'<p class="header-title">Admin Dashboard 👨‍💼</p>', unsafe_allow_html=True)
    
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM applications WHERE status='Pending'")
    pending_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM applications")
    total_count = c.fetchone()[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='loan-card'>
            <h2 style='color: #F39C12;'>⏳ {pending_count} Pending Applications</h2>
            <p style='color: #34495E;'>Review and process pending loan applications</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='loan-card'>
            <h2 style='color: #4A90E2;'>📊 {total_count} Total Applications</h2>
            <p style='color: #34495E;'>Total applications in the system</p>
        </div>
        """, unsafe_allow_html=True)

def show_pending_applications(conn):
    st.markdown('<p class="header-title">⏳ Pending Applications</p>', unsafe_allow_html=True)
    
    c = conn.cursor()
    c.execute("SELECT * FROM applications WHERE status='Pending' ORDER BY created_at DESC")
    pending = c.fetchall()
    
    if not pending:
        st.success("✅ No pending applications!")
        return
    
    for loan in pending:
        st.markdown(f"""
        <div class='loan-card'>
            <h3>👤 {loan[2]}</h3>
            <p><strong>Loan Amount:</strong> ${loan[11]:,.0f}</p>
            <p><strong>Credit Score:</strong> {loan[19]}</p>
            <p><strong>Annual Income:</strong> ${loan[10]:,.0f}</p>
            <p><strong>Employment:</strong> {loan[9]}</p>
            <p><strong>Applied On:</strong> {loan[22]}</p>
            <p><strong>Purpose:</strong> {loan[20]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"🤖 Run ML Model", key=f"ml_{loan[0]}"):
                data = {
                    'credit_score': loan[19],
                    'income': loan[10],
                    'loan_amount': loan[11],
                    'residential_assets': loan[13],
                    'commercial_assets': loan[14],
                    'luxury_assets': loan[15],
                    'bank_assets': loan[16],
                    'existing_loans': loan[18]
                }
                prediction = predict_loan(data)
                suggested_amount = calculate_suggested_loan_amount(data)
                
                st.session_state[f'prediction_{loan[0]}'] = prediction
                st.session_state[f'suggested_amount_{loan[0]}'] = suggested_amount
                
        # Display ML prediction if available
        if f'prediction_{loan[0]}' in st.session_state:
            prediction = st.session_state[f'prediction_{loan[0]}']
            suggested_amount = st.session_state.get(f'suggested_amount_{loan[0]}', 0)
            
            if prediction == "Approved":
                st.markdown(f"""
                <div style='background-color: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 1rem 0;'>
                    <h4 style='color: #155724; margin: 0 0 0.5rem 0;'>🤖 ML Model Recommendation: APPROVED ✅</h4>
                    <p style='color: #155724; margin: 0;'><strong>Requested Amount:</strong> ${loan[11]:,.0f}</p>
                    <p style='color: #155724; margin: 0;'><strong>Suggested Loan Amount:</strong> ${suggested_amount:,.0f}</p>
                    <p style='color: #0c5460; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>💡 The applicant's financial profile supports this loan. The suggested amount is based on income, credit score, and assets.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f8d7da; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545; margin: 1rem 0;'>
                    <h4 style='color: #721c24; margin: 0 0 0.5rem 0;'>🤖 ML Model Recommendation: REJECTED ❌</h4>
                    <p style='color: #721c24; margin: 0;'><strong>Requested Amount:</strong> ${loan[11]:,.0f}</p>
                    <p style='color: #721c24; margin: 0;'><strong>Suggested Maximum Amount:</strong> ${suggested_amount:,.0f}</p>
                    <p style='color: #856404; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>⚠️ The requested amount exceeds the applicant's capacity. Consider approving a lower amount of ${suggested_amount:,.0f} instead.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Approve only if ML predicted Approved
            if f'prediction_{loan[0]}' in st.session_state:
                prediction = st.session_state[f'prediction_{loan[0]}']
                if prediction == "Approved":
                    if st.button(f"✅ Approve", key=f"approve_{loan[0]}"):
                        c.execute("UPDATE applications SET status='Approved' WHERE id=?", (loan[0],))
                        conn.commit()
                        st.success("✅ Application approved!")
                        st.rerun()
                else:
                    st.button(f"✅ Approve", key=f"approve_disabled_{loan[0]}", disabled=True)
                    st.caption("🔒 ML predicted REJECTION")
            else:
                st.button(f"✅ Approve", key=f"approve_no_ml_{loan[0]}", disabled=True)
                st.caption("⚠️ Run ML Model first")
        
        with col3:
            # Reject only if ML predicted Rejected
            if f'prediction_{loan[0]}' in st.session_state:
                prediction = st.session_state[f'prediction_{loan[0]}']
                if prediction == "Rejected":
                    if st.button(f"❌ Reject", key=f"reject_{loan[0]}"):
                        c.execute("UPDATE applications SET status='Rejected' WHERE id=?", (loan[0],))
                        conn.commit()
                        st.error("❌ Application rejected!")
                        st.rerun()
                else:
                    st.button(f"❌ Reject", key=f"reject_disabled_{loan[0]}", disabled=True)
                    st.caption("🔒 ML predicted APPROVAL")
            else:
                st.button(f"❌ Reject", key=f"reject_no_ml_{loan[0]}", disabled=True)
                st.caption("⚠️ Run ML Model first")
        
        st.markdown("---")
def show_all_applications(conn):
    st.markdown('<p class="header-title">📋 All Applications</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female", "Other"])
    with col2:
        credit_filter = st.number_input("Min Credit Score", min_value=0, value=0)
    with col3:
        status_filter = st.selectbox("Filter by Status", ["All", "Pending", "Approved", "Rejected"])
    with col4:
        loan_filter = st.number_input("Min Loan Amount", min_value=0.0, value=0.0)
    
    c = conn.cursor()
    query = "SELECT * FROM applications WHERE 1=1"
    params = []
    
    if gender_filter != "All":
        query += " AND gender=?"
        params.append(gender_filter)
    if credit_filter > 0:
        query += " AND credit_score>=?"
        params.append(credit_filter)
    if status_filter != "All":
        query += " AND status=?"
        params.append(status_filter)
    if loan_filter > 0:
        query += " AND loan_amount>=?"
        params.append(loan_filter)
    
    query += " ORDER BY created_at DESC"
    
    c.execute(query, params)
    apps = c.fetchall()
    
    if apps:
        # Create comprehensive table with all 19 fields
        table_data = []
        for app in apps:
            table_data.append({
                'ID': app[0],
                'Full Name': app[2],  # app[2] is full_name field
                'Date of Birth': app[3],  # app[3] is dob field
                'Gender': app[4],
                'Email': app[5],
                'Phone': app[6],
                'Address': app[7],
                'Education': app[8],
                'Employment Type': app[9],
                'Annual Income': f"${app[10]:,.2f}",
                'Loan Amount': f"${app[11]:,.2f}",
                'Loan Term (months)': app[12],
                'Residential Assets': f"${app[13]:,.2f}",
                'Commercial Assets': f"${app[14]:,.2f}",
                'Luxury Assets': f"${app[15]:,.2f}",
                'Bank Assets': f"${app[16]:,.2f}",
                'Years of Employment': app[17],
                'Existing Loans': f"${app[18]:,.2f}",
                'Credit Score': app[19],
                'Loan Purpose': app[20],
                'Status': app[21],
                'Created': app[22]
            })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Add custom CSS for white background table
        st.markdown("""
        <style>
        /* Force white background for dataframe */
        div[data-testid="stDataFrame"] {
            background-color: white !important;
        }
        
        div[data-testid="stDataFrame"] > div {
            background-color: white !important;
        }
        
        div[data-testid="stDataFrame"] iframe {
            background-color: white !important;
        }
        
        /* Table header styling */
        .stDataFrame thead tr th {
            background-color: #f8f9fa !important;
            color: #2C3E50 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #dee2e6 !important;
        }
        
        /* Table body styling */
        .stDataFrame tbody tr td {
            color: #000000 !important;
            background-color: white !important;
            border-bottom: 1px solid #dee2e6 !important;
        }
        
        /* Hover effect */
        .stDataFrame tbody tr:hover td {
            background-color: #f8f9fa !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display comprehensive table
        st.dataframe(
            df,
            use_container_width=True,
            height=500,
            hide_index=True
        )
        
        st.info(f"📊 Showing {len(apps)} applications")
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"loan_applications_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    else:
        st.info("🔍 No applications match the selected filters.")

def show_admin_analytics(conn):
    st.markdown('<p class="header-title">📊 Loan Applicant Profile Dashboard</p>', unsafe_allow_html=True)
    
    c = conn.cursor()
    c.execute("SELECT * FROM applications")
    all_apps = c.fetchall()
    
    if not all_apps:
        st.info("📭 No data available yet.")
        return
    
    # Filters Section
    st.markdown("### 🔍 Filters")
    col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])
    
    with col1:
        gender_filter = st.selectbox("Gender:", ["All", "Male", "Female", "Other"])
    with col2:
        education_filter = st.selectbox("Education:", ["All", "High School", "Bachelor's", "Master's", "PhD", "Other"])
    with col3:
        purpose_filter = st.selectbox("Loan Purpose:", ["All", "Business", "Debt Consolidation", "Education", 
                                                         "Home Improvement", "Home Purchase", "Medical", "Personal", "Vehicle"])
    with col4:
        approval_filter = st.selectbox("Predicted Approval:", ["All", "Approved", "Rejected", "Pending"])
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)
        apply_filter = st.button("Apply Filters", use_container_width=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        min_credit = st.number_input("Min Credit Score:", min_value=300, max_value=850, value=300, step=10)
    
    # Apply filters
    filtered_apps = all_apps
    if gender_filter != "All":
        filtered_apps = [a for a in filtered_apps if a[4] == gender_filter]
    if education_filter != "All":
        filtered_apps = [a for a in filtered_apps if a[8] == education_filter]
    if purpose_filter != "All":
        filtered_apps = [a for a in filtered_apps if a[20] == purpose_filter]
    if approval_filter != "All":
        filtered_apps = [a for a in filtered_apps if a[21] == approval_filter]
    if min_credit > 300:
        filtered_apps = [a for a in filtered_apps if a[19] >= min_credit]
    
    # Calculate metrics
    total = len(filtered_apps)
    approved = len([a for a in filtered_apps if a[21] == "Approved"])
    rejected = len([a for a in filtered_apps if a[21] == "Rejected"])
    pending = len([a for a in filtered_apps if a[21] == "Pending"])
    
    avg_loan = sum([a[11] for a in filtered_apps]) / total if total > 0 else 0
    avg_income = sum([a[10] for a in filtered_apps]) / total if total > 0 else 0
    avg_credit = sum([a[19] for a in filtered_apps]) / total if total > 0 else 0
    approval_rate = (approved / total * 100) if total > 0 else 0
    
    st.markdown("---")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>📊</div>
            <h2 style='color: #4A90E2; margin: 0.5rem 0;'>{total}</h2>
            <p style='color: #34495E; font-weight: 600; margin: 0;'>Total Applications</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>💰</div>
            <h2 style='color: #4A90E2; margin: 0.5rem 0;'>${avg_loan:,.0f}</h2>
            <p style='color: #34495E; font-weight: 600; margin: 0;'>Average Loan Amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>💵</div>
            <h2 style='color: #4A90E2; margin: 0.5rem 0;'>${avg_income:,.0f}</h2>
            <p style='color: #34495E; font-weight: 600; margin: 0;'>Average Income</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>📈</div>
            <h2 style='color: #4A90E2; margin: 0.5rem 0;'>{avg_credit:.1f}</h2>
            <p style='color: #34495E; font-weight: 600; margin: 0;'>Average Credit Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 2rem;'>✅</div>
            <h2 style='color: #27AE60; margin: 0.5rem 0;'>{approval_rate:.1f}%</h2>
            <p style='color: #34495E; font-weight: 600; margin: 0;'>Approval Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # First Row of Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution by Gender
        age_data = []
        for app in filtered_apps:
            try:
                from datetime import datetime
                dob = datetime.strptime(app[3], '%Y-%m-%d')
                age = (datetime.now() - dob).days // 365
                age_data.append({'Age': age, 'Gender': app[4]})
            except:
                pass
        
        if age_data:
            age_df = pd.DataFrame(age_data)
            age_df['Age Range'] = pd.cut(age_df['Age'], bins=[0, 20, 30, 40, 50, 60, 70, 100], 
                                        labels=['≤ 20', '21 - 30', '31 - 40', '41 - 50', '51 - 60', '61 - 70', '71 - 80'])
            
            fig_age = px.histogram(age_df, x='Age Range', color='Gender', barmode='group',
                                  title='Age Distribution by Gender',
                                  color_discrete_map={'Male': '#5DADE2', 'Female': '#F1948A', 'Other': '#F7DC6F'})
            fig_age.update_layout(
                height=350, 
                plot_bgcolor='white', 
                paper_bgcolor='white',
                font=dict(color='#000000', size=12),
                title_font=dict(color='#2C3E50', size=16),
                xaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
                yaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
                legend=dict(font=dict(color='#000000'))
            )
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("No age data available")
    
    with col2:
        # Approval Probability Distribution
        approval_probs = []
        for app in filtered_apps:
            data = {
                'credit_score': app[19],
                'income': app[10],
                'loan_amount': app[11],
                'residential_assets': app[13],
                'commercial_assets': app[14],
                'luxury_assets': app[15],
                'bank_assets': app[16],
                'existing_loans': app[18]
            }
            # Calculate probability score
            score = 0
            if data['credit_score'] >= 750: score += 40
            elif data['credit_score'] >= 650: score += 25
            elif data['credit_score'] >= 550: score += 10
            
            if data['income'] >= 100000: score += 30
            elif data['income'] >= 50000: score += 20
            elif data['income'] >= 30000: score += 10
            
            approval_probs.append(score)
        
        if approval_probs:
            prob_df = pd.DataFrame({'Probability': approval_probs})
            prob_df['Range'] = pd.cut(prob_df['Probability'], 
                                     bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 100],
                                     labels=['0%-9%', '10%-19%', '20%-29%', '30%-39%', '40%-49%', 
                                            '50%-59%', '60%-69%', '70%-79%', '80%-89%', '90%-99%'])
            
            prob_counts = prob_df['Range'].value_counts().sort_index()
            
            # Create gradient colors from pink to green
            colors = ['#E91E63', '#EC407A', '#F06292', '#F48FB1', '#CE93D8', 
                     '#B39DDB', '#9FA8DA', '#90CAF9', '#81C784', '#66BB6A']
            bar_colors = [colors[i] if i < len(colors) else '#66BB6A' for i in range(len(prob_counts))]
            
            fig_prob = go.Figure(data=[go.Bar(x=prob_counts.index, y=prob_counts.values, 
                                             marker_color=bar_colors)])
            fig_prob.update_layout(
                title='Approval Probability Distribution',
                xaxis_title='Approval Probability',
                yaxis_title='Number of Applications',
                height=350, 
                plot_bgcolor='white', 
                paper_bgcolor='white',
                font=dict(color='#000000', size=12),
                title_font=dict(color='#2C3E50', size=16),
                xaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
                yaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000'))
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("No probability data available")
    
    # Second Row of Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Job Experience vs Income (Bubble Chart)
        scatter_data = []
        for app in filtered_apps:
            scatter_data.append({
                'Job Years': app[17],
                'Income': app[10],
                'Loan Amount': app[11],
                'Status': app[21]
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        fig_scatter = px.scatter(scatter_df, x='Job Years', y='Income', size='Loan Amount',
                               color='Status', title='Job Experience vs Income',
                               color_discrete_map={'Approved': '#66BB6A', 'Rejected': '#EF5350', 'Pending': '#FFA726'},
                               size_max=30)
        fig_scatter.update_layout(
            height=350, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            font=dict(color='#000000', size=12),
            title_font=dict(color='#2C3E50', size=16),
            xaxis=dict(
                title='Years of Job Experience',
                title_font=dict(color='#000000', size=12),
                tickfont=dict(color='#000000', size=10)
            ),
            yaxis=dict(
                title='Income ($)',
                title_font=dict(color='#000000', size=12),
                tickfont=dict(color='#000000', size=10)
            ),
            legend=dict(font=dict(color='#000000'))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Asset Distribution by Approval Status (Radar Chart)
        approved_apps = [a for a in filtered_apps if a[21] == 'Approved']
        rejected_apps = [a for a in filtered_apps if a[21] == 'Rejected']
        
        avg_res_approved = sum([a[13] for a in approved_apps]) / len(approved_apps) if approved_apps else 0
        avg_com_approved = sum([a[14] for a in approved_apps]) / len(approved_apps) if approved_apps else 0
        avg_lux_approved = sum([a[15] for a in approved_apps]) / len(approved_apps) if approved_apps else 0
        avg_bank_approved = sum([a[16] for a in approved_apps]) / len(approved_apps) if approved_apps else 0
        
        avg_res_rejected = sum([a[13] for a in rejected_apps]) / len(rejected_apps) if rejected_apps else 0
        avg_com_rejected = sum([a[14] for a in rejected_apps]) / len(rejected_apps) if rejected_apps else 0
        avg_lux_rejected = sum([a[15] for a in rejected_apps]) / len(rejected_apps) if rejected_apps else 0
        avg_bank_rejected = sum([a[16] for a in rejected_apps]) / len(rejected_apps) if rejected_apps else 0
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[avg_res_approved, avg_com_approved, avg_bank_approved, avg_lux_approved],
            theta=['Residential', 'Commercial', 'Bank', 'Luxury'],
            fill='toself',
            name='Approved',
            line_color='#66BB6A',
            fillcolor='rgba(102, 187, 106, 0.3)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[avg_res_rejected, avg_com_rejected, avg_bank_rejected, avg_lux_rejected],
            theta=['Residential', 'Commercial', 'Bank', 'Luxury'],
            fill='toself',
            name='Rejected',
            line_color='#EF5350',
            fillcolor='rgba(239, 83, 80, 0.3)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    tickfont=dict(color='#000000', size=10),
                    gridcolor='#D3D3D3'
                ),
                angularaxis=dict(
                    tickfont=dict(color='#000000', size=11),
                    gridcolor='#D3D3D3'
                ),
                bgcolor='white'
            ),
            title='Asset Distribution by Approval Status',
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#000000', size=12),
            title_font=dict(color='#2C3E50', size=16),
            legend=dict(font=dict(color='#000000'))
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Third Row of Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit Score vs Approval Probability
        credit_data = []
        for app in filtered_apps:
            data = {
                'credit_score': app[19],
                'income': app[10],
                'loan_amount': app[11],
                'residential_assets': app[13],
                'commercial_assets': app[14],
                'luxury_assets': app[15],
                'bank_assets': app[16],
                'existing_loans': app[18]
            }
            score = 0
            if data['credit_score'] >= 750: score += 40
            elif data['credit_score'] >= 650: score += 25
            elif data['credit_score'] >= 550: score += 10
            if data['income'] >= 100000: score += 30
            elif data['income'] >= 50000: score += 20
            elif data['income'] >= 30000: score += 10
            
            credit_data.append({
                'Credit Score': app[19],
                'Approval Probability': score,
                'Status': app[21]
            })
        
        credit_df = pd.DataFrame(credit_data)
        fig_credit = px.scatter(credit_df, x='Credit Score', y='Approval Probability',
                              color='Status', title='Credit Score vs Approval Probability',
                              color_discrete_map={'Approved': '#26C6DA', 'Rejected': '#8E8E93', 'Pending': '#FFA726'})
        fig_credit.update_layout(
            height=350, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            font=dict(color='#000000', size=12),
            title_font=dict(color='#2C3E50', size=16),
            xaxis=dict(
                title='Credit Score',
                title_font=dict(color='#000000', size=12),
                tickfont=dict(color='#000000', size=10)
            ),
            yaxis=dict(
                title='Approval Probability (%)',
                title_font=dict(color='#000000', size=12),
                tickfont=dict(color='#000000', size=10)
            ),
            legend=dict(font=dict(color='#000000'))
        )
        st.plotly_chart(fig_credit, use_container_width=True)
    
    with col2:
        # Monthly Payment by Approval Status
        payment_data = []
        for app in filtered_apps:
            # Calculate estimated monthly payment
            loan_amount = app[11]
            term_months = app[12]
            # Simple calculation: loan amount / term (ignoring interest for simplicity)
            monthly_payment = loan_amount / term_months if term_months > 0 else 0
            payment_data.append({
                'Monthly Payment': monthly_payment,
                'Status': app[21]
            })
        
        payment_df = pd.DataFrame(payment_data)
        payment_df['Payment Range'] = pd.cut(payment_df['Monthly Payment'],
                                            bins=[0, 499, 999, 1499, 1999, 2499, 2999, 3499, 3999, 4499, 10000],
                                            labels=['$0-$499', '$500-$999', '$1000-$1499', '$1500-$1999',
                                                   '$2000-$2499', '$2500-$2999', '$3000-$3499', '$3500-$3999',
                                                   '$4000-$4499', '$4500+'])
        
        payment_grouped = payment_df.groupby(['Payment Range', 'Status']).size().reset_index(name='Count')
        fig_payment = px.bar(payment_grouped, x='Payment Range', y='Count', color='Status',
                           title='Monthly Payment by Approval Status', barmode='group',
                           color_discrete_map={'Approved': '#66BB6A', 'Rejected': '#EF5350', 'Pending': '#FFA726'})
        fig_payment.update_layout(
            height=350, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            xaxis_tickangle=-45,
            font=dict(color='#000000', size=12),
            title_font=dict(color='#2C3E50', size=16),
            xaxis=dict(
                title='Estimated Monthly Payment',
                title_font=dict(color='#000000', size=12),
                tickfont=dict(color='#000000', size=9)
            ),
            yaxis=dict(
                title='Number of Applications',
                title_font=dict(color='#000000', size=12),
                tickfont=dict(color='#000000', size=10)
            ),
            legend=dict(font=dict(color='#000000'))
        )
        st.plotly_chart(fig_payment, use_container_width=True)

# ==================== MAIN APPLICATION ====================

def main():
    st.set_page_config(page_title="Loan Approval System", 
                      page_icon="🏦",
                      layout="wide",
                      initial_sidebar_state="auto")
    
    apply_custom_css()
    
    conn = init_db()
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        if st.session_state.page == 'home':
            show_home_page()
        elif st.session_state.page == 'register':
            show_register_page(conn)
        elif st.session_state.page == 'login':
            show_login_page(conn)
    else:
        if st.session_state.role == 'Applicant':
            show_applicant_dashboard(conn)
        elif st.session_state.role == 'Admin':
            show_admin_dashboard(conn)

if __name__ == "__main__":
    main()







