"""
Preprocessing module for email text cleaning and feature extraction
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class EmailPreprocessor:
    """
    Class untuk preprocessing email text
    """
    
    def __init__(self, use_stemming=True, remove_stopwords=True):
        """
        Initialize preprocessor
        
        Args:
            use_stemming (bool): Apakah menggunakan stemming
            remove_stopwords (bool): Apakah menghapus stopwords
        """
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_html(self, text):
        """
        Remove HTML tags from text
        
        Args:
            text (str): Input text with HTML
            
        Returns:
            str: Text without HTML tags
        """
        if not isinstance(text, str):
            return ""
        
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    
    def remove_urls(self, text):
        """
        Remove URLs from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without URLs
        """
        # Remove http/https URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        return text
    
    def remove_emails(self, text):
        """
        Remove email addresses from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without email addresses
        """
        text = re.sub(r'\S+@\S+', '', text)
        return text
    
    def remove_special_characters(self, text):
        """
        Remove special characters and numbers
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with only letters and spaces
        """
        # Keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return text
    
    def to_lowercase(self, text):
        """
        Convert text to lowercase
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lowercase text
        """
        return text.lower()
    
    def remove_extra_whitespace(self, text):
        """
        Remove extra whitespaces
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without extra whitespaces
        """
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords_from_tokens(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens without stopwords
        """
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        stemmed = [self.stemmer.stem(word) for word in tokens]
        return stemmed
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Cleaned and preprocessed text
        """
        # Step 1: Clean HTML
        text = self.clean_html(text)
        
        # Step 2: Remove URLs
        text = self.remove_urls(text)
        
        # Step 3: Remove email addresses
        text = self.remove_emails(text)
        
        # Step 4: Convert to lowercase
        text = self.to_lowercase(text)
        
        # Step 5: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 6: Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Step 7: Tokenize
        tokens = self.tokenize(text)
        
        # Step 8: Remove stopwords (optional)
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Step 9: Stemming (optional)
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)
        
        # Join tokens back to string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text


def extract_email_features(text):
    """
    Extract features from email text for phishing detection
    
    Args:
        text (str): Email text
        
    Returns:
        dict: Dictionary of features
    """
    features = {}
    
    # Feature 1: Email length
    features['text_length'] = len(text)
    
    # Feature 2: Number of URLs
    features['num_urls'] = len(re.findall(r'http\S+|www\S+|https\S+', text))
    
    # Feature 3: Number of email addresses
    features['num_emails'] = len(re.findall(r'\S+@\S+', text))
    
    # Feature 4: Number of special characters
    features['num_special_chars'] = sum(1 for char in text if char in string.punctuation)
    
    # Feature 5: Contains suspicious words
    suspicious_words = ['urgent', 'verify', 'account', 'suspended', 'click', 
                       'confirm', 'password', 'update', 'banking', 'security',
                       'prize', 'winner', 'claim', 'congratulations']
    
    text_lower = text.lower()
    features['num_suspicious_words'] = sum(1 for word in suspicious_words if word in text_lower)
    
    # Feature 6: All caps words count
    words = text.split()
    features['num_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1)
    
    # Feature 7: Has IP address
    features['has_ip'] = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text) else 0
    
    # Feature 8: Number of exclamation marks
    features['num_exclamation'] = text.count('!')
    
    # Feature 9: Number of question marks
    features['num_question'] = text.count('?')
    
    return features


def load_and_preprocess_data(filepath, text_column='email_text', label_column='label'):
    """
    Load dataset and preprocess all emails
    
    Args:
        filepath (str): Path to CSV file
        text_column (str): Name of column containing email text
        label_column (str): Name of column containing labels
        
    Returns:
        tuple: (preprocessed_texts, labels, features_df)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"Dataset loaded: {len(df)} emails")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize preprocessor
    preprocessor = EmailPreprocessor(use_stemming=True, remove_stopwords=True)
    
    # Preprocess texts
    print("Preprocessing emails...")
    df['cleaned_text'] = df[text_column].apply(preprocessor.preprocess)
    
    # Extract features
    print("Extracting features...")
    features_list = df[text_column].apply(extract_email_features)
    features_df = pd.DataFrame(features_list.tolist())
    
    print("Preprocessing complete!")
    
    return df['cleaned_text'].values, df[label_column].values, features_df


if __name__ == "__main__":
    # Test preprocessing
    sample_email = """
    <html>
    <body>
    URGENT: Your account has been suspended!
    
    Dear Customer,
    
    We have detected suspicious activity on your account. 
    Please click here http://phishing-site.com to verify your account immediately.
    
    Your account will be permanently closed if you don't respond within 24 hours!
    
    Contact us at support@fake-bank.com
    
    Thank you,
    Security Team
    </body>
    </html>
    """
    
    # Initialize preprocessor
    preprocessor = EmailPreprocessor()
    
    # Clean email
    cleaned = preprocessor.preprocess(sample_email)
    print("Original Email:")
    print(sample_email)
    print("\n" + "="*50 + "\n")
    print("Cleaned Email:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Extract features
    features = extract_email_features(sample_email)
    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
