"""
Flask Web Application for Phishing Email Detector
Provides web interface and REST API for phishing detection
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import PhishingDetector

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize detector
try:
    detector = PhishingDetector()
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Warning: Could not load model - {e}")
    MODEL_LOADED = False


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', model_loaded=MODEL_LOADED)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for prediction
    
    Request JSON:
    {
        "email_text": "Your email content here"
    }
    
    Response JSON:
    {
        "success": true,
        "prediction": {
            "is_phishing": true/false,
            "label": "Phishing" or "Legitimate",
            "confidence": 0.85,
            "confidence_percent": 85.0
        }
    }
    """
    try:
        # Check if model is loaded
        if not MODEL_LOADED:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 503
        
        # Get email text from request
        data = request.get_json()
        
        if not data or 'email_text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing email_text in request'
            }), 400
        
        email_text = data['email_text']
        
        if not email_text or not email_text.strip():
            return jsonify({
                'success': False,
                'error': 'Email text cannot be empty'
            }), 400
        
        # Make prediction
        result = detector.predict(email_text)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    API endpoint for detailed analysis
    
    Response includes prediction + features
    """
    try:
        if not MODEL_LOADED:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        data = request.get_json()
        
        if not data or 'email_text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing email_text in request'
            }), 400
        
        email_text = data['email_text']
        
        if not email_text or not email_text.strip():
            return jsonify({
                'success': False,
                'error': 'Email text cannot be empty'
            }), 400
        
        # Detailed analysis
        analysis = detector.analyze_email(email_text)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch prediction
    
    Request JSON:
    {
        "emails": ["email1", "email2", "email3"]
    }
    """
    try:
        if not MODEL_LOADED:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        data = request.get_json()
        
        if not data or 'emails' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing emails array in request'
            }), 400
        
        emails = data['emails']
        
        if not isinstance(emails, list):
            return jsonify({
                'success': False,
                'error': 'emails must be an array'
            }), 400
        
        if len(emails) > 100:
            return jsonify({
                'success': False,
                'error': 'Maximum 100 emails per batch'
            }), 400
        
        # Batch prediction
        results = detector.predict_batch(emails)
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED
    })


@app.route('/api/info')
def api_info():
    """API information"""
    return jsonify({
        'name': 'Phishing Email Detector API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'Web interface (GET)',
            '/predict': 'Single email prediction (POST)',
            '/analyze': 'Detailed email analysis (POST)',
            '/batch': 'Batch prediction (POST)',
            '/health': 'Health check (GET)',
            '/api/info': 'API information (GET)'
        },
        'model_loaded': MODEL_LOADED
    })


@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 PHISHING EMAIL DETECTOR - WEB APPLICATION")
    print("="*80)
    print(f"\nModel Status: {'✅ Loaded' if MODEL_LOADED else '❌ Not Loaded'}")
    
    if not MODEL_LOADED:
        print("\n⚠️  WARNING: Model not loaded!")
        print("   Please train the model first: python train_model.py")
    
    print("\n🌐 Starting server...")
    print("   Local URL: http://127.0.0.1:5000")
    print("   Network URL: http://0.0.0.0:5000")
    print("\n📚 API Endpoints:")
    print("   GET  /              - Web interface")
    print("   POST /predict       - Single prediction")
    print("   POST /analyze       - Detailed analysis")
    print("   POST /batch         - Batch prediction")
    print("   GET  /health        - Health check")
    print("   GET  /api/info      - API info")
    print("\n💡 Press CTRL+C to stop the server")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
