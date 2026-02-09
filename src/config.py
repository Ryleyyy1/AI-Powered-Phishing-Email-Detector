"""
Configuration file for phishing email detector
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebooks')

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, 'phishing_detector.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Data paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'phishing_emails.csv')

# Preprocessing parameters
PREPROCESSING_CONFIG = {
    'use_stemming': True,
    'remove_stopwords': True,
    'max_features': 5000,  # For TF-IDF vectorizer
    'ngram_range': (1, 2),  # Unigrams and bigrams
}

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'algorithm': 'naive_bayes',  # Options: 'naive_bayes', 'random_forest', 'svm'
}

# Random Forest specific parameters
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42,
}

# SVM specific parameters
SVM_CONFIG = {
    'kernel': 'linear',
    'C': 1.0,
    'random_state': 42,
}

# Flask app configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
}

# Feature extraction
SUSPICIOUS_WORDS = [
    'urgent', 'verify', 'account', 'suspended', 'click', 
    'confirm', 'password', 'update', 'banking', 'security',
    'prize', 'winner', 'claim', 'congratulations', 'free',
    'credit card', 'social security', 'act now', 'limited time',
    'expire', 'validation', 'authenticate', 'refund'
]
