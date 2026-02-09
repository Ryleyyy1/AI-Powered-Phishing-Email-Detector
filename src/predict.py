"""
Prediction Script for Phishing Email Detection
Load trained model and make predictions on new emails
"""

import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.preprocessing import EmailPreprocessor
except ImportError:
    from src.simple_preprocessor import SimpleEmailPreprocessor as EmailPreprocessor


class PhishingDetector:
    """
    Phishing email detector using trained model
    """
    
    def __init__(self, model_path='models/phishing_model.pkl', 
                 vectorizer_path='models/vectorizer.pkl'):
        """
        Initialize detector with trained model
        
        Args:
            model_path (str): Path to trained model
            vectorizer_path (str): Path to vectorizer
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and vectorizer"""
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            try:
                self.preprocessor = EmailPreprocessor(use_stemming=True, remove_stopwords=True)
            except:
                self.preprocessor = EmailPreprocessor()
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print("❌ Model files not found!")
            print("   Please train the model first: python train_model.py")
            raise
    
    def predict(self, email_text):
        """
        Predict if email is phishing
        
        Args:
            email_text (str): Email text to analyze
            
        Returns:
            dict: Prediction results
        """
        # Preprocess
        cleaned_text = self.preprocessor.preprocess(email_text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_vec)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 1.0  # SVM without probability
        
        result = {
            'is_phishing': bool(prediction),
            'label': 'Phishing' if prediction == 1 else 'Legitimate',
            'confidence': float(confidence),
            'confidence_percent': float(confidence * 100)
        }
        
        return result
    
    def predict_batch(self, emails):
        """
        Predict multiple emails
        
        Args:
            emails (list): List of email texts
            
        Returns:
            list: List of prediction results
        """
        return [self.predict(email) for email in emails]
    
    def analyze_email(self, email_text):
        """
        Detailed analysis of email
        
        Args:
            email_text (str): Email text
            
        Returns:
            dict: Detailed analysis
        """
        import re
        
        result = self.predict(email_text)
        
        # Additional analysis
        analysis = {
            'prediction': result,
            'features': {
                'length': len(email_text),
                'num_urls': len(re.findall(r'http\S+|www\S+|https\S+', email_text)),
                'num_emails': len(re.findall(r'\S+@\S+', email_text)),
                'num_exclamation': email_text.count('!'),
                'num_caps_words': sum(1 for word in email_text.split() 
                                     if word.isupper() and len(word) > 1),
                'has_urgency': any(word in email_text.lower() 
                                  for word in ['urgent', 'immediately', 'act now', 'hurry'])
            }
        }
        
        return analysis


def test_samples():
    """Test the detector with sample emails"""
    
    # Initialize detector
    detector = PhishingDetector()
    
    # Test samples
    test_emails = [
        {
            'text': "URGENT! Your account will be suspended! Click http://fake-bank.com/verify now to prevent closure!",
            'expected': 'Phishing'
        },
        {
            'text': "Thank you for your order #12345. Your package will arrive in 3-5 business days.",
            'expected': 'Legitimate'
        },
        {
            'text': "Congratulations! You won $1,000,000! Click here http://scam.com to claim your prize NOW!!!",
            'expected': 'Phishing'
        },
        {
            'text': "Meeting reminder: Team standup tomorrow at 10 AM. Please review the agenda.",
            'expected': 'Legitimate'
        }
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING PHISHING DETECTOR")
    print("="*80)
    
    correct = 0
    for i, sample in enumerate(test_emails, 1):
        print(f"\n📧 Test Email {i}:")
        print(f"   Text: {sample['text'][:60]}...")
        print(f"   Expected: {sample['expected']}")
        
        result = detector.predict(sample['text'])
        print(f"   Predicted: {result['label']}")
        print(f"   Confidence: {result['confidence_percent']:.2f}%")
        
        is_correct = (result['label'] == sample['expected'])
        correct += is_correct
        print(f"   {'✅ Correct' if is_correct else '❌ Wrong'}")
    
    print(f"\n{'='*80}")
    print(f"📊 Accuracy: {correct}/{len(test_emails)} ({correct/len(test_emails)*100:.1f}%)")
    print(f"{'='*80}")


def interactive_mode():
    """Interactive mode for testing emails"""
    
    detector = PhishingDetector()
    
    print("\n" + "="*80)
    print("🎯 INTERACTIVE PHISHING DETECTOR")
    print("="*80)
    print("\nEnter email text (or 'quit' to exit)")
    print("Type 'analyze' for detailed analysis")
    print("-"*80)
    
    while True:
        print("\n📧 Enter email text:")
        email_text = input("> ")
        
        if email_text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not email_text.strip():
            continue
        
        if email_text.lower() == 'analyze':
            print("\n📧 Enter email to analyze:")
            email_text = input("> ")
            analysis = detector.analyze_email(email_text)
            
            print("\n" + "="*80)
            print("📊 DETAILED ANALYSIS")
            print("="*80)
            
            pred = analysis['prediction']
            print(f"\n🎯 Prediction: {pred['label']}")
            print(f"   Confidence: {pred['confidence_percent']:.2f}%")
            
            print(f"\n📈 Features:")
            for feature, value in analysis['features'].items():
                print(f"   {feature}: {value}")
        else:
            result = detector.predict(email_text)
            
            print("\n" + "-"*80)
            if result['is_phishing']:
                print("🚨 WARNING: PHISHING EMAIL DETECTED!")
            else:
                print("✅ Email appears to be LEGITIMATE")
            
            print(f"   Confidence: {result['confidence_percent']:.2f}%")
            print("-"*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phishing Email Detector')
    parser.add_argument('--test', action='store_true', help='Run test samples')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--email', type=str, help='Single email to analyze')
    
    args = parser.parse_args()
    
    if args.test:
        test_samples()
    elif args.interactive:
        interactive_mode()
    elif args.email:
        detector = PhishingDetector()
        result = detector.predict(args.email)
        
        print(f"\n{'='*80}")
        print(f"🎯 Prediction: {result['label']}")
        print(f"   Is Phishing: {result['is_phishing']}")
        print(f"   Confidence: {result['confidence_percent']:.2f}%")
        print(f"{'='*80}")
    else:
        # Default: run test
        test_samples()
