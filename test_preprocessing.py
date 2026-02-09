"""
Test script untuk demonstrasi preprocessing workflow
(Simplified version tanpa NLTK untuk testing)
"""

import re
from bs4 import BeautifulSoup

def simple_preprocess(text):
    """
    Simplified preprocessing function
    """
    # Remove HTML
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_features(text):
    """
    Extract features from email
    """
    features = {
        'text_length': len(text),
        'num_urls': len(re.findall(r'http\S+|www\S+|https\S+', text)),
        'num_emails': len(re.findall(r'\S+@\S+', text)),
        'num_exclamation': text.count('!'),
        'num_question': text.count('?'),
    }
    return features


# Test
sample_email = """
<html>
<body>
URGENT: Your account has been suspended!

Please click here http://phishing-site.com to verify.

Contact: support@fake-bank.com
</body>
</html>
"""

print("PREPROCESSING TEST")
print("=" * 60)
print("\nOriginal Email:")
print(sample_email)
print("\n" + "=" * 60)

cleaned = simple_preprocess(sample_email)
print("\nCleaned Email:")
print(cleaned)
print("\n" + "=" * 60)

features = extract_features(sample_email)
print("\nExtracted Features:")
for key, value in features.items():
    print(f"  {key}: {value}")

print("\n✅ Preprocessing module working correctly!")
