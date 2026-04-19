"""
PAN Card Verification Module
Implements PAN validation and mock credit bureau integration
"""

import re
import random
from datetime import datetime, timedelta

# ============================================================================
# LEVEL 1: PAN CARD FORMAT VALIDATION
# ============================================================================

def validate_pan_format(pan_number):
    """
    Validates PAN card format according to Indian standards
    
    Format: AAAAA9999A
    - First 5: Alphabets (uppercase)
    - Next 4: Numbers
    - Last 1: Alphabet (uppercase)
    
    Returns: (is_valid, error_message)
    """
    if not pan_number:
        return False, "PAN number is required"
    
    # Remove spaces and convert to uppercase
    pan_number = pan_number.replace(" ", "").upper()
    
    # Check length
    if len(pan_number) != 10:
        return False, "PAN must be 10 characters long"
    
    # Validate format using regex
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
    if not re.match(pattern, pan_number):
        return False, "Invalid PAN format. Format should be: AAAAA9999A"
    
    # Check 4th character (indicates entity type)
    entity_types = {
        'P': 'Individual',
        'C': 'Company',
        'H': 'HUF (Hindu Undivided Family)',
        'F': 'Firm/Partnership',
        'A': 'Association of Persons',
        'T': 'Trust',
        'B': 'Body of Individuals',
        'L': 'Local Authority',
        'J': 'Artificial Juridical Person',
        'G': 'Government'
    }
    
    fourth_char = pan_number[3]
    if fourth_char not in entity_types:
        return False, f"Invalid entity type code: {fourth_char}"
    
    # For loan applications, typically we expect 'P' (Individual)
    entity_type = entity_types[fourth_char]
    
    return True, f"Valid PAN ({entity_type})"


def extract_pan_info(pan_number):
    """
    Extract information encoded in PAN number
    """
    pan_number = pan_number.replace(" ", "").upper()
    
    info = {
        'pan_number': pan_number,
        'entity_type_code': pan_number[3],
        'entity_type': {
            'P': 'Individual',
            'C': 'Company',
            'H': 'HUF',
            'F': 'Firm',
            'A': 'AOP',
            'T': 'Trust'
        }.get(pan_number[3], 'Unknown'),
        'is_individual': pan_number[3] == 'P'
    }
    
    return info


# ============================================================================
# LEVEL 2: MOCK CREDIT BUREAU API (For Demonstration)
# ============================================================================

class MockCreditBureau:
    """
    Simulates credit bureau API responses
    In production, replace with actual CIBIL/Experian/Equifax API
    """
    
    def __init__(self):
        self.known_pans = self._generate_sample_database()
    
    def _generate_sample_database(self):
        """Generate sample PAN records for testing"""
        return {
            'ABCDE1234F': {
                'name': 'John Doe',
                'dob': '1985-05-15',
                'credit_score': 780,
                'total_accounts': 5,
                'active_loans': 2,
                'total_debt': 350000,
                'defaulter': False,
                'last_updated': '2024-02-15'
            },
            'PQRST5678G': {
                'name': 'Jane Smith',
                'dob': '1990-08-22',
                'credit_score': 620,
                'total_accounts': 3,
                'active_loans': 1,
                'total_debt': 180000,
                'defaulter': True,
                'last_updated': '2024-01-20'
            }
        }
    
    def verify_pan(self, pan_number, name=None, dob=None):
        """
        Mock PAN verification
        
        Returns:
        {
            'status': 'success' | 'error',
            'verified': True | False,
            'credit_data': {...} | None,
            'message': str
        }
        """
        pan_number = pan_number.replace(" ", "").upper()
        
        # Simulate API delay
        import time
        time.sleep(0.5)
        
        # Check if PAN exists in database
        if pan_number in self.known_pans:
            record = self.known_pans[pan_number]
            
            # Verify name if provided
            name_match = True
            if name:
                # Simple name matching (in real API, this would be more sophisticated)
                name_match = name.lower() in record['name'].lower() or \
                            record['name'].lower() in name.lower()
            
            # Verify DOB if provided
            dob_match = True
            if dob:
                dob_match = dob == record['dob']
            
            if name_match and dob_match:
                return {
                    'status': 'success',
                    'verified': True,
                    'credit_data': record,
                    'message': 'PAN verified successfully'
                }
            else:
                return {
                    'status': 'error',
                    'verified': False,
                    'credit_data': None,
                    'message': 'PAN exists but name/DOB mismatch'
                }
        else:
            # Generate synthetic credit data for demo purposes
            synthetic_score = random.randint(550, 850)
            
            return {
                'status': 'success',
                'verified': True,
                'credit_data': {
                    'name': name or 'Not Available',
                    'dob': dob or 'Not Available',
                    'credit_score': synthetic_score,
                    'total_accounts': random.randint(1, 8),
                    'active_loans': random.randint(0, 3),
                    'total_debt': random.randint(0, 500000),
                    'defaulter': synthetic_score < 600,
                    'last_updated': datetime.now().strftime('%Y-%m-%d')
                },
                'message': 'PAN verified (simulated data for demo)'
            }
    
    def get_credit_score(self, pan_number):
        """Get just the credit score"""
        result = self.verify_pan(pan_number)
        if result['verified'] and result['credit_data']:
            return result['credit_data']['credit_score']
        return None
    
    def check_defaulter_status(self, pan_number):
        """Check if PAN holder is a defaulter"""
        result = self.verify_pan(pan_number)
        if result['verified'] and result['credit_data']:
            return result['credit_data'].get('defaulter', False)
        return None


# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def verify_pan_and_fetch_credit(pan_number, applicant_name=None, dob=None):
    """
    Complete PAN verification workflow
    
    Steps:
    1. Validate PAN format
    2. Verify with credit bureau
    3. Fetch credit score and financial data
    4. Return comprehensive result
    """
    
    # Step 1: Format validation
    is_valid, validation_msg = validate_pan_format(pan_number)
    
    if not is_valid:
        return {
            'success': False,
            'error': validation_msg,
            'credit_score': None,
            'credit_data': None
        }
    
    # Step 2: Extract PAN info
    pan_info = extract_pan_info(pan_number)
    
    # Only individuals can apply for personal loans
    if not pan_info['is_individual']:
        return {
            'success': False,
            'error': f"PAN is registered as {pan_info['entity_type']}, not Individual",
            'credit_score': None,
            'credit_data': None
        }
    
    # Step 3: Verify with credit bureau (mock)
    bureau = MockCreditBureau()
    verification_result = bureau.verify_pan(pan_number, applicant_name, dob)
    
    if not verification_result['verified']:
        return {
            'success': False,
            'error': verification_result['message'],
            'credit_score': None,
            'credit_data': None
        }
    
    # Step 4: Return complete data
    credit_data = verification_result['credit_data']
    
    return {
        'success': True,
        'message': verification_result['message'],
        'pan_number': pan_number,
        'credit_score': credit_data['credit_score'],
        'credit_data': {
            'name': credit_data['name'],
            'dob': credit_data['dob'],
            'credit_score': credit_data['credit_score'],
            'total_accounts': credit_data['total_accounts'],
            'active_loans': credit_data['active_loans'],
            'total_debt': credit_data['total_debt'],
            'is_defaulter': credit_data['defaulter'],
            'last_updated': credit_data['last_updated']
        }
    }


# ============================================================================
# CREDIT SCORE RISK ASSESSMENT
# ============================================================================

def assess_credit_risk(credit_score):
    """
    Assess risk level based on credit score
    """
    if credit_score >= 750:
        return {
            'risk_level': 'Low',
            'category': 'Excellent',
            'approval_recommendation': 'Highly Recommended',
            'interest_rate_category': 'Prime (Lowest)',
            'color': 'green'
        }
    elif credit_score >= 700:
        return {
            'risk_level': 'Low-Medium',
            'category': 'Good',
            'approval_recommendation': 'Recommended',
            'interest_rate_category': 'Standard',
            'color': 'lightgreen'
        }
    elif credit_score >= 650:
        return {
            'risk_level': 'Medium',
            'category': 'Fair',
            'approval_recommendation': 'Moderate Risk',
            'interest_rate_category': 'Standard-High',
            'color': 'orange'
        }
    elif credit_score >= 600:
        return {
            'risk_level': 'Medium-High',
            'category': 'Below Average',
            'approval_recommendation': 'High Risk',
            'interest_rate_category': 'High',
            'color': 'darkorange'
        }
    else:
        return {
            'risk_level': 'High',
            'category': 'Poor',
            'approval_recommendation': 'Not Recommended',
            'interest_rate_category': 'Very High / Decline',
            'color': 'red'
        }


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_pan_verification():
    """Test the PAN verification system"""
    
    print("="*60)
    print("PAN VERIFICATION SYSTEM TEST")
    print("="*60)
    
    test_cases = [
        {
            'pan': 'ABCDE1234F',
            'name': 'John Doe',
            'dob': '1985-05-15'
        },
        {
            'pan': 'INVALID123',
            'name': 'Test User',
            'dob': '1990-01-01'
        },
        {
            'pan': 'PQRST5678G',
            'name': 'Jane Smith',
            'dob': '1990-08-22'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"PAN: {test['pan']}")
        print(f"Name: {test['name']}")
        print(f"DOB: {test['dob']}")
        
        result = verify_pan_and_fetch_credit(
            test['pan'],
            test['name'],
            test['dob']
        )
        
        print(f"\nResult:")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"Credit Score: {result['credit_score']}")
            print(f"Active Loans: {result['credit_data']['active_loans']}")
            print(f"Total Debt: ₹{result['credit_data']['total_debt']:,}")
            print(f"Defaulter: {result['credit_data']['is_defaulter']}")
            
            risk = assess_credit_risk(result['credit_score'])
            print(f"\nRisk Assessment:")
            print(f"  Category: {risk['category']}")
            print(f"  Risk Level: {risk['risk_level']}")
            print(f"  Recommendation: {risk['approval_recommendation']}")
        else:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    test_pan_verification()

