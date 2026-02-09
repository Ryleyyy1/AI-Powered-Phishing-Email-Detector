import nltk
import ssl

# Fix SSL certificate error (jika ada)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download resources
nltk.download('punkt_tab')
nltk.download('stopwords')
print("✅ NLTK resources downloaded successfully!")