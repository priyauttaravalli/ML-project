import pickle
import os
import numpy as np
import base64
from flask import Flask, request, render_template, redirect
import tensorflow as tf
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Configure TensorFlow for compatibility
tf.config.run_functions_eagerly(True)
# Enable mixed precision to match the saved model
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Apply ResNet50V2 preprocessing
    return img_array

# Initialize Flask app
app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model from pickle file
try:
    # Try to import FocalLoss if available
    from keras_cv.losses import FocalLoss
except ImportError:
    # Define a simple replacement if keras_cv is not available
    class FocalLoss:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, y_true, y_pred):
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

print("Loading model from pickle file...")

# Try multiple approaches to load the model
model = None
try:
    # Approach 1: Load with float32 precision
    tf.keras.mixed_precision.set_global_policy('float32')
    custom_objects = {'FocalLoss': FocalLoss}
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        with open('resnet50v2_waste_classifier.pkl', 'rb') as file:
            model = pickle.load(file)
    print("Model loaded successfully with float32!")
    
except Exception as e1:
    print(f"Failed to load with float32: {e1}")
    
    try:
        # Approach 2: Try loading without mixed precision at all
        with open('resnet50v2_waste_classifier.pkl', 'rb') as file:
            model_data = pickle.load(file)
            # Try to extract just the weights if possible
            if hasattr(model_data, 'get_weights'):
                print("Model structure loaded, attempting to use weights...")
                model = model_data
            else:
                model = model_data
        print("Model loaded with alternative approach!")
        
    except Exception as e2:
        print(f"All loading approaches failed: {e2}")
        print("The model file appears to be corrupted or incompatible.")
        model = None

# Define image preprocessing function


# Define categories
categories = ['Glass', 'Shoes', 'Clothes', 'Trash', 'Plastic', 'Paper', 'Metal', 'Cardboard', 'Biological', 'Battery']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/env')
def env():
    return render_template('evironment.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if an image was uploaded via file input
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
    
    # Check if an image was captured from the camera
    elif 'capturedImage' in request.form and request.form['capturedImage'].startswith('data:image/png;base64'):
        # Decode base64 string
        image_data = request.form['capturedImage'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Save the image
        filepath = os.path.join(UPLOAD_FOLDER, 'captured_image.png')
        image.save(filepath)
    
    else:
        return redirect('/')

    # Preprocess and predict
    if model is None:
        return render_template('result.html', filename=os.path.basename(filepath), prediction="Model Loading Error")
    
    try:
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]
        
        return render_template('result.html', filename=os.path.basename(filepath), prediction=predicted_class)
    except Exception as e:
        return render_template('result.html', filename=os.path.basename(filepath), prediction=f"Prediction Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
