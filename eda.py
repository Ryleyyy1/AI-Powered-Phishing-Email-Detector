"""
Exploratory Data Analysis (EDA) untuk Phishing Email Dataset
"""

import pandas as pd
import re
from collections import Counter


def analyze_dataset(filepath='data/sample_emails.csv'):
    """
    Perform comprehensive EDA on email dataset
    
    Args:
        filepath (str): Path to CSV file
    """
    # Load data
    df = pd.read_csv(filepath)
    
    print("="*80)
    print("📊 PHISHING EMAIL DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Basic Info
    print(f"\n{'='*80}")
    print("1️⃣ BASIC INFORMATION")
    print(f"{'='*80}")
    print(f"Total Emails: {len(df)}")
    print(f"Phishing Emails: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    print(f"Legitimate Emails: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    
    # Text Length Analysis
    print(f"\n{'='*80}")
    print("2️⃣ TEXT LENGTH ANALYSIS")
    print(f"{'='*80}")
    
    df['text_length'] = df['email_text'].apply(len)
    
    print("\nPhishing Emails:")
    phishing = df[df['label'] == 1]
    print(f"  Average length: {phishing['text_length'].mean():.0f} characters")
    print(f"  Min length: {phishing['text_length'].min()}")
    print(f"  Max length: {phishing['text_length'].max()}")
    
    print("\nLegitimate Emails:")
    legitimate = df[df['label'] == 0]
    print(f"  Average length: {legitimate['text_length'].mean():.0f} characters")
    print(f"  Min length: {legitimate['text_length'].min()}")
    print(f"  Max length: {legitimate['text_length'].max()}")
    
    # URL Analysis
    print(f"\n{'='*80}")
    print("3️⃣ URL ANALYSIS")
    print(f"{'='*80}")
    
    def count_urls(text):
        return len(re.findall(r'http\S+|www\S+|https\S+', str(text)))
    
    df['num_urls'] = df['email_text'].apply(count_urls)
    
    phishing = df[df['label'] == 1]
    legitimate = df[df['label'] == 0]
    
    print("\nPhishing Emails:")
    print(f"  Emails with URLs: {sum(phishing['num_urls'] > 0)} ({sum(phishing['num_urls'] > 0)/len(phishing)*100:.1f}%)")
    print(f"  Average URLs per email: {phishing['num_urls'].mean():.2f}")
    
    print("\nLegitimate Emails:")
    print(f"  Emails with URLs: {sum(legitimate['num_urls'] > 0)} ({sum(legitimate['num_urls'] > 0)/len(legitimate)*100:.1f}%)")
    print(f"  Average URLs per email: {legitimate['num_urls'].mean():.2f}")
    
    # Suspicious Keywords Analysis
    print(f"\n{'='*80}")
    print("4️⃣ SUSPICIOUS KEYWORDS ANALYSIS")
    print(f"{'='*80}")
    
    suspicious_words = [
        'urgent', 'verify', 'account', 'suspended', 'click', 
        'confirm', 'password', 'update', 'security', 'prize',
        'winner', 'claim', 'congratulations', 'warning', 'alert'
    ]
    
    def count_suspicious_words(text):
        text_lower = str(text).lower()
        return sum(1 for word in suspicious_words if word in text_lower)
    
    df['suspicious_count'] = df['email_text'].apply(count_suspicious_words)
    
    phishing = df[df['label'] == 1]
    legitimate = df[df['label'] == 0]
    
    print("\nPhishing Emails:")
    print(f"  Average suspicious words: {phishing['suspicious_count'].mean():.2f}")
    print(f"  Emails with suspicious words: {sum(phishing['suspicious_count'] > 0)}/{len(phishing)}")
    
    print("\nLegitimate Emails:")
    print(f"  Average suspicious words: {legitimate['suspicious_count'].mean():.2f}")
    print(f"  Emails with suspicious words: {sum(legitimate['suspicious_count'] > 0)}/{len(legitimate)}")
    
    # Most Common Words in Phishing Emails
    print(f"\n{'='*80}")
    print("5️⃣ TOP WORDS IN PHISHING EMAILS")
    print(f"{'='*80}")
    
    phishing_text = ' '.join(phishing['email_text'].values).lower()
    words = re.findall(r'\b[a-z]+\b', phishing_text)
    word_freq = Counter(words).most_common(15)
    
    print("\nTop 15 words in phishing emails:")
    for word, count in word_freq:
        print(f"  {word:15s}: {count:3d}")
    
    # Exclamation Marks Analysis
    print(f"\n{'='*80}")
    print("6️⃣ PUNCTUATION ANALYSIS")
    print(f"{'='*80}")
    
    df['num_exclamation'] = df['email_text'].apply(lambda x: str(x).count('!'))
    df['num_question'] = df['email_text'].apply(lambda x: str(x).count('?'))
    
    phishing = df[df['label'] == 1]
    legitimate = df[df['label'] == 0]
    
    print("\nExclamation Marks (!):")
    print(f"  Phishing avg: {phishing['num_exclamation'].mean():.2f}")
    print(f"  Legitimate avg: {legitimate['num_exclamation'].mean():.2f}")
    
    print("\nQuestion Marks (?):")
    print(f"  Phishing avg: {phishing['num_question'].mean():.2f}")
    print(f"  Legitimate avg: {legitimate['num_question'].mean():.2f}")
    
    # ALL CAPS Analysis
    print(f"\n{'='*80}")
    print("7️⃣ ALL CAPS WORDS ANALYSIS")
    print(f"{'='*80}")
    
    def count_caps_words(text):
        words = str(text).split()
        return sum(1 for word in words if word.isupper() and len(word) > 1)
    
    df['caps_words'] = df['email_text'].apply(count_caps_words)
    
    phishing = df[df['label'] == 1]
    legitimate = df[df['label'] == 0]
    
    print(f"\nPhishing emails with ALL CAPS words: {sum(phishing['caps_words'] > 0)}/{len(phishing)}")
    print(f"Legitimate emails with ALL CAPS words: {sum(legitimate['caps_words'] > 0)}/{len(legitimate)}")
    
    # Sample Emails
    print(f"\n{'='*80}")
    print("8️⃣ SAMPLE EMAILS")
    print(f"{'='*80}")
    
    print("\n🚨 PHISHING EMAIL SAMPLE:")
    print("-" * 80)
    print(phishing['email_text'].iloc[0])
    
    print("\n✅ LEGITIMATE EMAIL SAMPLE:")
    print("-" * 80)
    print(legitimate['email_text'].iloc[0])
    
    print(f"\n{'='*80}")
    print("✅ EDA COMPLETE!")
    print(f"{'='*80}\n")
    
    return df


if __name__ == "__main__":
    # Run EDA on sample dataset
    df = analyze_dataset('data/sample_emails.csv')
    
    print("\n💡 Key Insights:")
    print("  • Phishing emails tend to have more URLs")
    print("  • Phishing emails use more suspicious keywords")
    print("  • Phishing emails have more exclamation marks")
    print("  • Phishing emails often use ALL CAPS words")
    print("  • These patterns can be used as features for ML model")
