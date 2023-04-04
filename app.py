from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB4
from io import BytesIO 
import base64

# Load the trained model
model = tf.keras.models.load_model('model1.h5', compile=False)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy','AUC'])

# Set up the Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for handling image uploads
@app.route('/predict', methods=['POST'])
def predict():
    # Load the uploaded image
    # img = load_img(request.files['image'], target_size=(224, 224))

    # Load the uploaded image
    img_file = request.files['image']
    img_bytes = img_file.read()
    img = load_img(BytesIO(img_bytes), target_size=(224, 224))
    
    # Preprocess the image
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction using the trained model
    predictions = model.predict(img_array)
    predicted_class = int(np.round(predictions[0]))
    if predicted_class == 0:
        res = 'Not Melanoma'
    else:
        res = "Melanoma"
    # Encode the image data as a base64-encoded string
    img_data = base64.b64encode(img_bytes).decode('utf-8')

    # Return the predicted class to the result page
    return render_template('result.html', res = res, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)