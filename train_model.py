"""
Model Training Script for Phishing Email Detection
Supports multiple algorithms: Naive Bayes, Random Forest, SVM, Logistic Regression
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Import preprocessing module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.preprocessing import EmailPreprocessor
except ImportError:
    # Fallback to simple preprocessor if NLTK not available
    from src.simple_preprocessor import SimpleEmailPreprocessor as EmailPreprocessor
    print("⚠️  Using simplified preprocessor (NLTK not available)")


class PhishingDetectorTrainer:
    """
    Trainer class untuk phishing email detector
    """
    
    def __init__(self, algorithm='naive_bayes', max_features=5000):
        """
        Initialize trainer
        
        Args:
            algorithm (str): 'naive_bayes', 'random_forest', 'svm', 'logistic_regression'
            max_features (int): Maximum features for TF-IDF
        """
        self.algorithm = algorithm
        self.max_features = max_features
        self.vectorizer = None
        self.model = None
        try:
            self.preprocessor = EmailPreprocessor(use_stemming=True, remove_stopwords=True)
        except:
            self.preprocessor = EmailPreprocessor()  # Simple version without params
        
    def load_data(self, filepath='data/sample_emails.csv'):
        """
        Load and preprocess dataset
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            tuple: (X, y)
        """
        print(f"\n{'='*80}")
        print("📂 LOADING DATASET")
        print(f"{'='*80}")
        
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} emails")
        print(f"   Phishing: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
        print(f"   Legitimate: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
        
        # Preprocess texts
        print("\n🔄 Preprocessing emails...")
        df['cleaned_text'] = df['email_text'].apply(self.preprocessor.preprocess)
        
        X = df['cleaned_text'].values
        y = df['label'].values
        
        print("✅ Preprocessing complete!")
        
        return X, y
    
    def create_model(self):
        """
        Create model based on algorithm choice
        
        Returns:
            model: Scikit-learn model
        """
        if self.algorithm == 'naive_bayes':
            return MultinomialNB()
        
        elif self.algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.algorithm == 'svm':
            return SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=42
            )
        
        elif self.algorithm == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model
        
        Args:
            X (array): Feature array
            y (array): Labels
            test_size (float): Test set size
            
        Returns:
            dict: Training results
        """
        print(f"\n{'='*80}")
        print(f"🤖 TRAINING MODEL: {self.algorithm.upper()}")
        print(f"{'='*80}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n📊 Dataset Split:")
        print(f"   Training set: {len(X_train)} emails")
        print(f"   Test set: {len(X_test)} emails")
        
        # Vectorize text
        print(f"\n🔤 Creating TF-IDF features (max_features={self.max_features})...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,
            max_df=0.95
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"✅ Feature matrix shape: {X_train_vec.shape}")
        
        # Create and train model
        print(f"\n⚙️ Training {self.algorithm}...")
        self.model = self.create_model()
        self.model.fit(X_train_vec, y_train)
        print("✅ Training complete!")
        
        # Predictions
        print("\n🎯 Making predictions...")
        y_pred_train = self.model.predict(X_train_vec)
        y_pred_test = self.model.predict(X_test_vec)
        
        # Get probabilities for AUC
        if hasattr(self.model, 'predict_proba'):
            y_proba_test = self.model.predict_proba(X_test_vec)[:, 1]
        else:
            y_proba_test = self.model.decision_function(X_test_vec)
        
        # Evaluate
        results = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test),
            'auc': roc_auc_score(y_test, y_proba_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, 
                                                          target_names=['Legitimate', 'Phishing'])
        }
        
        # Cross-validation
        print("\n🔄 Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_vec, y_train, cv=5)
        results['cv_scores'] = cv_scores
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        return results
    
    def print_results(self, results):
        """
        Print training results
        
        Args:
            results (dict): Training results
        """
        print(f"\n{'='*80}")
        print("📈 MODEL PERFORMANCE")
        print(f"{'='*80}")
        
        print(f"\n🎯 Accuracy:")
        print(f"   Training: {results['train_accuracy']:.4f} ({results['train_accuracy']*100:.2f}%)")
        print(f"   Testing:  {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        
        print(f"\n📊 Metrics:")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall:    {results['recall']:.4f}")
        print(f"   F1-Score:  {results['f1']:.4f}")
        print(f"   AUC-ROC:   {results['auc']:.4f}")
        
        print(f"\n🔄 Cross-Validation (5-fold):")
        print(f"   Mean accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
        print(f"   Individual folds: {[f'{score:.4f}' for score in results['cv_scores']]}")
        
        print(f"\n🎭 Confusion Matrix:")
        cm = results['confusion_matrix']
        print(f"                 Predicted")
        print(f"                 Legit  Phish")
        print(f"   Actual Legit   {cm[0][0]:3d}    {cm[0][1]:3d}")
        print(f"          Phish   {cm[1][0]:3d}    {cm[1][1]:3d}")
        
        print(f"\n📋 Classification Report:")
        print(results['classification_report'])
    
    def save_model(self, model_dir='models'):
        """
        Save trained model and vectorizer
        
        Args:
            model_dir (str): Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(model_dir, f'{self.algorithm}_{timestamp}.pkl')
        vectorizer_path = os.path.join(model_dir, f'vectorizer_{timestamp}.pkl')
        
        # Also save as default (for easy loading)
        default_model_path = os.path.join(model_dir, 'phishing_detector.pkl')
        default_vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        # Save models
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.model, default_model_path)
        joblib.dump(self.vectorizer, default_vectorizer_path)
        
        print(f"\n{'='*80}")
        print("💾 SAVING MODELS")
        print(f"{'='*80}")
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Vectorizer saved: {vectorizer_path}")
        print(f"✅ Default model saved: {default_model_path}")
        print(f"✅ Default vectorizer saved: {default_vectorizer_path}")


def compare_algorithms(filepath='data/sample_emails.csv'):
    """
    Compare multiple algorithms
    
    Args:
        filepath (str): Path to dataset
    """
    algorithms = ['naive_bayes', 'logistic_regression', 'random_forest', 'svm']
    results_comparison = {}
    
    print(f"\n{'='*80}")
    print("🔬 COMPARING MULTIPLE ALGORITHMS")
    print(f"{'='*80}")
    
    for algo in algorithms:
        print(f"\n{'='*80}")
        print(f"Testing: {algo.upper()}")
        print(f"{'='*80}")
        
        trainer = PhishingDetectorTrainer(algorithm=algo, max_features=3000)
        X, y = trainer.load_data(filepath)
        results = trainer.train(X, y)
        trainer.print_results(results)
        
        results_comparison[algo] = {
            'accuracy': results['test_accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc']
        }
    
    # Print comparison
    print(f"\n{'='*80}")
    print("📊 ALGORITHM COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Algorithm':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 80)
    
    for algo, metrics in results_comparison.items():
        print(f"{algo:<20} {metrics['accuracy']:.4f}    {metrics['precision']:.4f}     "
              f"{metrics['recall']:.4f}    {metrics['f1']:.4f}    {metrics['auc']:.4f}")
    
    # Find best algorithm
    best_algo = max(results_comparison.items(), key=lambda x: x[1]['f1'])
    print(f"\n🏆 Best Algorithm: {best_algo[0].upper()} (F1-Score: {best_algo[1]['f1']:.4f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Phishing Email Detector')
    parser.add_argument('--algorithm', type=str, default='naive_bayes',
                       choices=['naive_bayes', 'random_forest', 'svm', 'logistic_regression'],
                       help='Algorithm to use')
    parser.add_argument('--dataset', type=str, default='data/sample_emails.csv',
                       help='Path to dataset')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all algorithms')
    parser.add_argument('--max-features', type=int, default=5000,
                       help='Max features for TF-IDF')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare all algorithms
        compare_algorithms(args.dataset)
    else:
        # Train single algorithm
        print(f"\n{'='*80}")
        print("🎯 PHISHING EMAIL DETECTOR - MODEL TRAINING")
        print(f"{'='*80}")
        
        trainer = PhishingDetectorTrainer(
            algorithm=args.algorithm,
            max_features=args.max_features
        )
        
        X, y = trainer.load_data(args.dataset)
        results = trainer.train(X, y)
        trainer.print_results(results)
        trainer.save_model()
        
        print(f"\n{'='*80}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*80}")
        print("\nNext steps:")
        print("  1. Test the model: python src/predict.py")
        print("  2. Run the web app: python app.py")
