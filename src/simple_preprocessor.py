"""
Simplified Email Preprocessor (without NLTK dependency)
For environments where NLTK is not available
"""

import re
from bs4 import BeautifulSoup


class SimpleEmailPreprocessor:
    """
    Simplified preprocessor without NLTK
    """
    
    def __init__(self):
        # Common English stopwords (simplified list)
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
    
    def clean_html(self, text):
        """Remove HTML tags"""
        if not isinstance(text, str):
            return ""
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text()
        except:
            return text
    
    def remove_urls(self, text):
        """Remove URLs"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        return text
    
    def remove_emails(self, text):
        """Remove email addresses"""
        text = re.sub(r'\S+@\S+', '', text)
        return text
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Cleaned text
        """
        # Clean HTML
        text = self.clean_html(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove emails
        text = self.remove_emails(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords (optional - simple version)
        words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        return ' '.join(words)


if __name__ == "__main__":
    # Test
    preprocessor = SimpleEmailPreprocessor()
    
    sample = """
    <html>URGENT! Your account has been suspended. 
    Click http://fake.com to verify. Contact support@fake.com</html>
    """
    
    cleaned = preprocessor.preprocess(sample)
    print("Original:", sample)
    print("\nCleaned:", cleaned)
