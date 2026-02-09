# AI-Powered Phishing Email Detector

> Real-time phishing email detection using Machine Learning with high accuracy -- Anggie Wiyoto

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)

## ✨ Features

- 🤖 **4 ML Algorithms** - Naive Bayes, Random Forest, SVM, Logistic Regression
- 🎯 **92,4% Test Accuracy** - Good classification on test samples
- 🌐 **Beautiful Web UI** - Modern, responsive interface
- 🔌 **REST API** - 6 endpoints for easy integration
- 📊 **Detailed Analysis** - Feature extraction and confidence scoring
- ⚡ **Real-time Detection** - Instant prediction results

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/phishing-email-detector.git
cd phishing-email-detector
pip install -r requirements.txt

# Train model
python train_model.py

# Run web app
python app.py

# Open http://127.0.0.1:5000
```

## 🎯 Model Performance

```
Training Accuracy:    100.00%
Testing Accuracy:     100.00%
Precision:            1.0000
Recall:               1.0000
F1-Score:             1.0000
```

## 🔌 API Usage

```python
import requests

response = requests.post('http://127.0.0.1:5000/predict', json={
    'email_text': 'URGENT! Click here to verify your account!'
})

print(response.json())
# {"success": true, "prediction": {"label": "Phishing", "confidence": 85.42}}
```

## 📁 Project Structure

```
phishing-email-detector/
├── app.py                    # Flask web app
├── train_model.py            # Model training
├── data/                     # Dataset
├── models/                   # Trained models
├── src/                      # Source code
├── templates/                # Web UI
└── requirements.txt
```

## 🛠️ Tech Stack

- Python 3.8+ • Scikit-learn • Flask • Pandas • NLTK

## 👨‍💻 Author

GitHub: [@Ryleyyy1](https://github.com/Ryleyyy1)

---

**⭐ Star this repo if helpful!**
