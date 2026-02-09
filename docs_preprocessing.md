# Preprocessing Module Documentation

## Overview

Module `preprocessing.py` bertanggung jawab untuk membersihkan dan memproses text email sebelum digunakan untuk training atau prediksi.

## Komponen Utama

### 1. EmailPreprocessor Class

Class utama untuk preprocessing email dengan berbagai metode pembersihan.

#### Methods:

- **`clean_html(text)`**: Menghapus HTML tags
- **`remove_urls(text)`**: Menghapus URL dari text
- **`remove_emails(text)`**: Menghapus alamat email
- **`remove_special_characters(text)`**: Menghapus karakter special
- **`to_lowercase(text)`**: Convert ke lowercase
- **`remove_extra_whitespace(text)`**: Menghapus whitespace berlebih
- **`tokenize(text)`**: Memecah text menjadi tokens
- **`remove_stopwords_from_tokens(tokens)`**: Menghapus stopwords
- **`stem_tokens(tokens)`**: Apply stemming
- **`preprocess(text)`**: Complete preprocessing pipeline

### 2. Feature Extraction

Function `extract_email_features()` mengekstrak fitur-fitur penting:

1. **text_length**: Panjang email
2. **num_urls**: Jumlah URL dalam email
3. **num_emails**: Jumlah alamat email
4. **num_special_chars**: Jumlah karakter special
5. **num_suspicious_words**: Jumlah kata-kata mencurigakan
6. **num_caps_words**: Jumlah kata ALL CAPS
7. **has_ip**: Apakah mengandung IP address
8. **num_exclamation**: Jumlah tanda seru
9. **num_question**: Jumlah tanda tanya

## Suspicious Words List

Kata-kata yang sering muncul di phishing emails:
- urgent, verify, account, suspended
- click, confirm, password, update
- banking, security, prize, winner
- claim, congratulations, dan lain-lain

## Usage Example

```python
from src.preprocessing import EmailPreprocessor, extract_email_features

# Initialize preprocessor
preprocessor = EmailPreprocessor(
    use_stemming=True,
    remove_stopwords=True
)

# Clean email
email_text = "Your sample email here..."
cleaned_text = preprocessor.preprocess(email_text)

# Extract features
features = extract_email_features(email_text)
print(features)
```

## Pipeline Preprocessing

1. Clean HTML tags
2. Remove URLs
3. Remove email addresses
4. Convert to lowercase
5. Remove special characters
6. Remove extra whitespace
7. Tokenization
8. Remove stopwords (optional)
9. Stemming (optional)

## Dependencies

- `nltk`: Natural Language Processing
- `beautifulsoup4`: HTML parsing
- `pandas`: Data manipulation
- `re`: Regular expressions

## Testing

Run the test:
```bash
python src/preprocessing.py
```

Ini akan menjalankan test dengan sample phishing email.
