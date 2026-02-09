"""
API Testing Script for Phishing Email Detector
Test all API endpoints
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*80)
    print("Testing /health endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_predict():
    """Test single prediction"""
    print("\n" + "="*80)
    print("Testing /predict endpoint")
    print("="*80)
    
    phishing_email = "URGENT! Your account will be suspended! Click here http://fake-bank.com"
    
    payload = {"email_text": phishing_email}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def run_all_tests():
    """Run all API tests"""
    print("\n🧪 PHISHING EMAIL DETECTOR - API TESTS")
    print("="*80)
    print(f"Base URL: {BASE_URL}")
    print("\nMake sure Flask server is running: python app.py\n")
    
    test_health()
    test_predict()


if __name__ == "__main__":
    run_all_tests()
