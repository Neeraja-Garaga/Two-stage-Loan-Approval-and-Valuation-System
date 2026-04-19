"""
Admin Account Creation Script
For administration use only - creates admin user accounts
"""

import sqlite3
import hashlib
import getpass

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_admin_account():
    """Create a new admin account"""
    print("="*60)
    print("🔐 ADMIN ACCOUNT CREATION")
    print("="*60)
    print("\nThis script creates administrator accounts.")
    print("Only authorized personnel should use this.\n")
    
    username = input("Enter admin username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return
    
    email = input("Enter admin email: ").strip()
    if not email or '@' not in email:
        print("❌ Invalid email address")
        return
    
    password = getpass.getpass("Enter admin password: ")
    confirm_password = getpass.getpass("Confirm password: ")
    
    if password != confirm_password:
        print("❌ Passwords do not match")
        return
    
    if len(password) < 8:
        print("❌ Password must be at least 8 characters")
        return
    
    try:
        conn = sqlite3.connect('loan_system.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE username=? OR email=?", (username, email))
        if cursor.fetchone():
            print("❌ Username or email already exists")
            conn.close()
            return
        
        hashed_password = hash_password(password)
        cursor.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (username, email, hashed_password, 'Admin'))
        
        conn.commit()
        conn.close()
        
        print("\n✅ Admin account created successfully!")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   Role: Admin")
        print("\n⚠️  Keep credentials secure!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def list_admin_accounts():
    """List all existing admin accounts"""
    try:
        conn = sqlite3.connect('loan_system.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT username, email FROM users WHERE role='Admin'")
        admins = cursor.fetchall()
        
        if admins:
            print("\n📋 Existing Admin Accounts:")
            print("-" * 60)
            for username, email in admins:
                print(f"   {username:20s} {email}")
        else:
            print("\n⚠️  No admin accounts found")
        
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    while True:
        print("\n" + "="*60)
        print("ADMIN MANAGEMENT")
        print("="*60)
        print("\n1. Create new admin account")
        print("2. List existing admin accounts")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            create_admin_account()
        elif choice == '2':
            list_admin_accounts()
        elif choice == '3':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()
