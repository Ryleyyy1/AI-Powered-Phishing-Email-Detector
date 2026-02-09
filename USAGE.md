# 🚀 Running the Web Application

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)

```bash
python train_model.py
```

This will create:
- `models/phishing_detector.pkl`
- `models/vectorizer.pkl`

### 3. Start the Flask Server

```bash
python app.py
```

The server will start at: **http://localhost:5000**

### 4. Access the Web Interface

Open your browser and navigate to:
- **Main App**: http://localhost:5000
- **API Docs**: http://localhost:5000/about

## 🎯 Using the Web Interface

1. **Paste Email Content**: Copy and paste any email text into the text area
2. **Click "Analyze Email"**: The AI will analyze the email
3. **View Results**: Get instant prediction with confidence score
4. **See Details**: View detailed analysis of email features

### Example Emails Included

The interface includes 4 pre-loaded examples:
- 2 Phishing emails
- 2 Legitimate emails

Click any example button to load it instantly!

## 🔌 Using the API

### Health Check

```bash
curl http://localhost:5000/api/health
```

### Single Email Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "email_text": "URGENT! Your account will be suspended!"
  }'
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "emails": [
      "First email text",
      "Second email text"
    ]
  }'
```

### Python Example

```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "email_text": "Your email content here"
}

response = requests.post(url, json=data)
result = response.json()

if result['success']:
    print(f"Prediction: {result['prediction']['label']}")
    print(f"Confidence: {result['prediction']['confidence_percent']:.2f}%")
```

## 🧪 Testing the API

Run the comprehensive API test suite:

```bash
# Make sure the server is running first!
python test_api.py
```

This will test:
- ✅ Health check endpoint
- ✅ Single prediction
- ✅ Batch prediction
- ✅ Error handling
- ✅ Performance

## 🐳 Docker Deployment (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t phishing-detector .
docker run -p 5000:5000 phishing-detector
```

## ☁️ Heroku Deployment

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## 📊 API Response Format

### Success Response

```json
{
  "success": true,
  "prediction": {
    "is_phishing": true,
    "label": "Phishing",
    "confidence": 0.8523,
    "confidence_percent": 85.23
  },
  "analysis": {
    "features": {
      "length": 156,
      "num_urls": 1,
      "num_emails": 1,
      "num_exclamation": 2,
      "num_caps_words": 3,
      "has_urgency": true
    }
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": "Error message here"
}
```

## ⚙️ Configuration

Edit `src/config.py` to customize:
- Model paths
- Server settings
- Feature extraction parameters

## 🔧 Troubleshooting

### Server won't start

**Error**: `Model files not found`

**Solution**: Train the model first
```bash
python train_model.py
```

### Port already in use

**Error**: `Address already in use`

**Solution**: Change port in `app.py` or kill the process
```bash
# Change port
app.run(port=5001)

# Or kill process
lsof -ti:5000 | xargs kill -9
```

### Import errors

**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

## 📱 Mobile Access

Access from mobile devices on the same network:

1. Find your computer's IP address:
```bash
# On Mac/Linux
ifconfig | grep inet

# On Windows
ipconfig
```

2. Access from mobile: `http://YOUR_IP:5000`

## 🔒 Security Notes

- This is a demo application
- For production use:
  - Add authentication
  - Use HTTPS
  - Implement rate limiting
  - Add input validation
  - Set up logging

## 📈 Performance Tips

- Use Gunicorn for production
- Enable caching for repeated predictions
- Use Redis for session management
- Deploy behind Nginx for better performance

## 🎨 Customization

### Change Colors

Edit CSS in `templates/index.html`:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Modify Features

Edit `src/predict.py` to add new features:
```python
def analyze_email(self, email_text):
    # Add your custom features here
    pass
```

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [API Testing with Postman](https://www.postman.com/)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - see LICENSE file for details
