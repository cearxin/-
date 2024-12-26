from pyngrok import ngrok
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import cv2
import numpy as np
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Flask app setup
app = Flask(__name__)

# Load pre-trained model from Google Drive
MODEL_PATH = '/content/drive/My Drive/my_model.h5'  # Update this path
model = tf.keras.models.load_model(MODEL_PATH)

ngrok.set_auth_token("2qk1n8upMxFXN43zARazUdcMlgP_57hSccrVE81EadE96HPLT")

# HTML template for the main page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digit Recognition API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        h1 {
            color: #333;
        }
        p {
            color: #555;
        }
        form {
            margin-top: 20px;
            padding: 20px;
            background-color: white;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .file-upload {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 20px;
            cursor: pointer;
            color: #888;
            margin-bottom: 10px;
            text-align: center;
        }
        .file-upload:hover {
            background-color: #f9f9f9;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        input[type="file"] {
            display: none;
        }
        img {
            margin-top: 10px;
            max-width: 100%;
            max-height: 200px;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <h1>Digit Recognition API</h1>
    <p>Upload or capture a digit image to get a prediction.</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <label class="file-upload" for="file">
            <span>Click here to upload or capture an image</span>
            <input id="file" type="file" name="file" accept="image/*" required onchange="previewImage(event)">
        </label>
        <img id="preview" src="#" alt="Image Preview" style="display: none;">
        <button type="submit">Predict</button>
    </form>
    {% if result is not none %}
    <h2>Predicted Digit: {{ result }}</h2>
    {% endif %}

    <script>
        function previewImage(event) {
            const file = event.target.files[0];
            const preview = document.getElementById('preview');
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            } else {
                preview.style.display = 'none';
            }
        }
    </script>
</body>
</html>
'''

# Flask routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE, result="Error: No file uploaded")
    
    file = request.files['file']
    image = preprocess_image(file.read())
    if image is None:
        return render_template_string(HTML_TEMPLATE, result="Error: Invalid image")
    
    prediction = model.predict(np.array([image]))[0]
    predicted_digit = int(np.argmax(prediction))
    return render_template_string(HTML_TEMPLATE, result=predicted_digit)

def preprocess_image(image_bytes, target_size=(28, 28)):
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, target_size)
        normalized_img = resized_img / 255.0
        normalized_img = np.expand_dims(normalized_img, axis=-1)
        return normalized_img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(" * ngrok tunnel available at:", public_url)
    
    # Start Flask app
    app.run()
