from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load your pretrained model
model = load_model('D:\\Project\\mini_project(5th_Sem)\\chest_xray_vgg16.h5')

# Directory to save uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files.get("image")
        if not image_file:
            return "No file uploaded!", 400

        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        try:
            # Preprocess the image
            image = load_img(image_path, target_size=(224, 224))  # Adjust to your model's input size
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) / 255.0  # Normalize if necessary

            # Predict using the model
            prediction = model.predict(image)[0]
            no_pneumonia_confidence = prediction[0] * 100  # No Pneumonia confidence
            pneumonia_confidence = prediction[1] * 100  # Pneumonia confidence
            result = "Pneumonia Detected" if pneumonia_confidence > 50 else "No Pneumonia"

            return render_template(
                "result.html",
                image_path=image_file.filename,  # Pass only the file name
                result=result,
                confidence=f"No Pneumonia: {no_pneumonia_confidence:.2f}%, Pneumonia: {pneumonia_confidence:.2f}%"
            )
        except Exception as e:
            return f"Error during prediction: {e}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
