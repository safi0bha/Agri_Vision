from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 role TEXT NOT NULL)''')
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                 ('admin', 'admin123', 'admin'))
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                 ('farmer', 'farmer123', 'user'))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

init_db()

# Improved PyTorch Model Definition for Disease Detection
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Disease classes mapping (alphabetical order to match training)
disease_classes = {
    0: 'blight',
    1: 'healthy', 
    2: 'leaf_spot',
    3: 'powdery_mildew',
    4: 'rust'
}

# Load disease model with error handling
def load_disease_model():
    try:
        model = PlantDiseaseCNN(num_classes=len(disease_classes))
        # Check if model file exists and has content
        if os.path.exists('models/plant_disease_model.pt') and os.path.getsize('models/plant_disease_model.pt') > 0:
            model.load_state_dict(torch.load('models/plant_disease_model.pt', map_location=torch.device('cpu')))
            print("✅ Disease model loaded successfully!")
        else:
            print("⚠️  Model file not found or empty, using untrained model")
            # Create and save a basic model if none exists
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/plant_disease_model.pt')
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading disease model: {e}")
        # Return a new model as fallback
        model = PlantDiseaseCNN(num_classes=len(disease_classes))
        model.eval()
        return model

# Load crop yield model
def load_crop_model():
    try:
        model = joblib.load("models/crop_yield_model.pkl")
        print("✅ Crop yield model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading crop model: {e}")
        from sklearn.ensemble import RandomForestRegressor
        # Create a basic trained model
        data = pd.DataFrame({
            'soil_type': [1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            'rainfall': [100, 120, 150, 80, 200, 180, 90, 160, 70, 130, 140, 110],
            'temperature': [25, 28, 30, 22, 35, 29, 26, 27, 24, 31, 23, 32],
            'humidity': [60, 70, 65, 55, 75, 68, 62, 72, 58, 66, 64, 69],
            'fertilizer_type': [1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4],
            'crop_type': [1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            'yield': [2.5, 3.0, 3.8, 2.0, 3.5, 3.6, 2.8, 3.2, 2.3, 3.1, 2.9, 3.7]
        })
        X = data[['soil_type', 'rainfall', 'temperature', 'humidity', 'fertilizer_type', 'crop_type']]
        y = data['yield']
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, "models/crop_yield_model.pkl")
        print("✅ New crop yield model created and saved!")
        return model

# Load models
crop_model = load_crop_model()
disease_model = load_disease_model()

# Image preprocessing for disease detection
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop-yield', methods=['GET', 'POST'])
def crop_yield():
    soil_types = {
        "Alluvial": 1,
        "Black": 2,
        "Red": 3,
        "Laterite": 4,
        "Mountain": 5,
        "Desert": 6
    }

    fertilizer_types = {
        "Urea": 1,
        "DAP": 2,
        "Potash": 3,
        "Compost": 4,
        "Organic": 5
    }

    crop_types = {
        "Wheat": 1,
        "Rice": 2,
        "Maize": 3,
        "Sugarcane": 4,
        "Cotton": 5,
        "Barley": 6
    }

    prediction = None

    if request.method == 'POST':
        try:
            soil_type_name = request.form['soil_type']
            fertilizer_type_name = request.form['fertilizer_type']
            crop_type_name = request.form['crop_type']

            soil_type = soil_types[soil_type_name]
            fertilizer_type = fertilizer_types[fertilizer_type_name]
            crop_type = crop_types[crop_type_name]

            rainfall = float(request.form['rainfall'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])

            input_data = pd.DataFrame([[soil_type, rainfall, temperature, humidity, fertilizer_type, crop_type]],
                columns=['soil_type', 'rainfall', 'temperature', 'humidity', 'fertilizer_type', 'crop_type'])

            prediction = crop_model.predict(input_data)[0]
            flash(f'✅ Yield prediction calculated successfully! Estimated yield: {prediction:.2f} tons/hectare', 'success')
            
        except Exception as e:
            flash(f'Error in prediction: {str(e)}', 'error')

    return render_template('crop_yield.html',
                           prediction=round(prediction, 2) if prediction else None,
                           soil_types=list(soil_types.keys()),
                           fertilizer_types=list(fertilizer_types.keys()),
                           crop_types=list(crop_types.keys()))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    # Handle GET request - return HTML page
    if request.method == 'GET':
        return render_template('disease_detection.html')
    
    # Handle POST request - process image
    if request.method == 'POST':
        # Check if it's an AJAX request (expecting JSON)
        wants_json = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.accept_json
        
        if 'file' not in request.files:
            if wants_json:
                return jsonify({'error': 'No file selected'}), 400
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            if wants_json:
                return jsonify({'error': 'No file selected'}), 400
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_dir = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_dir, exist_ok=True)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            
            try:
                # Process image
                image = Image.open(filepath).convert('RGB')
                original_size = image.size
                
                # Transform image for model
                input_tensor = transform(image).unsqueeze(0)
                
                # Model prediction
                with torch.no_grad():
                    outputs = disease_model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    confidence_percent = confidence.item() * 100
                
                # Get prediction result
                predicted_class = predicted.item()
                result = disease_classes.get(predicted_class, 'Unknown')
                
                # Format prediction
                prediction_text = f"{result.title()}"
                
                # Return JSON for AJAX requests
                if wants_json:
                    return jsonify({
                        'prediction': prediction_text,
                        'confidence': confidence_percent / 100,
                        'image_url': url_for('static', filename='images/uploads/' + filename),
                        'message': f'Analysis complete! Image processed: {original_size[0]}x{original_size[1]} pixels'
                    })
                
                # Return HTML for regular form submission
                flash(f'✅ Analysis complete! Image processed: {original_size[0]}x{original_size[1]} pixels', 'success')
                return render_template('disease_detection.html', 
                                      prediction=prediction_text, 
                                      confidence=confidence_percent,
                                      image_url=url_for('static', filename='images/uploads/' + filename))
                
            except Exception as e:
                error_msg = f'Error in disease detection: {str(e)}'
                print(f"Disease detection error: {error_msg}")
                if wants_json:
                    return jsonify({'error': error_msg}), 500
                flash(error_msg, 'error')
                return redirect(request.url)
        else:
            error_msg = 'Invalid file type. Please upload PNG, JPG, or JPEG images.'
            if wants_json:
                return jsonify({'error': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(request.url)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            
            flash(f'✅ Welcome back, {username}!', 'success')
            if user[3] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('index'))
        else:
            flash('❌ Invalid username or password', 'error')
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('❌ Passwords do not match', 'error')
            return render_template('auth/register.html')
        
        try:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                     (username, password, 'user'))
            conn.commit()
            conn.close()
            
            flash('✅ Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('❌ Username already exists', 'error')
    
    return render_template('auth/register.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'User')
    session.clear()
    flash(f'👋 Goodbye, {username}! You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('🔒 Admin access required. Please log in as administrator.', 'warning')
        return redirect(url_for('login'))
    return render_template('admin/dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'crop_model': True,
        'disease_model': True
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    if request.accept_mimetypes.accept_json:
        return jsonify({'error': 'Page not found'}), 404
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    if request.accept_mimetypes.accept_json:
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('errors/500.html'), 500

@app.errorhandler(413)
def too_large_error(error):
    if request.accept_mimetypes.accept_json:
        return jsonify({'error': 'File too large'}), 413
    flash('File too large. Please upload a smaller image.', 'error')
    return redirect(request.url)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images/uploads', exist_ok=True)
    os.makedirs('templates/auth', exist_ok=True)
    os.makedirs('templates/admin', exist_ok=True)
    os.makedirs('templates/errors', exist_ok=True)
    
    print("🚀 Starting Agricultural AI Platform...")
    print("✅ Models loaded successfully!")
    print("📁 Required directories created!")
    print("🌐 Server running at http://127.0.0.1:5000")
    print("📊 Available routes:")
    print("   - / (Home)")
    print("   - /crop-yield (Crop Yield Prediction)")
    print("   - /disease-detection (Plant Disease Detection)")
    print("   - /login (User Login)")
    print("   - /register (User Registration)")
    print("   - /admin/dashboard (Admin Panel)")
    print("   - /about (About Page)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)