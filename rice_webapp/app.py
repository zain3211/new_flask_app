from flask import Flask, request, redirect, url_for, render_template, send_from_directory, make_response
import os
import cv2
from werkzeug.utils import secure_filename
from histogram_equalization import histogram_equalization_rgb
from contrast_stretching import contrast_stretching_parabolic_rgb
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'
app.config['GRAPHS_FOLDER'] = 'static/graphs/'
processed_images = {}  # Define the processed_images dictionary at a higher scope

# Ensure the processed and graphs folders exist
for folder in [app.config['PROCESSED_FOLDER'], app.config['GRAPHS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load the models
model_original = tf.keras.models.load_model('models/first.h5')
model_he = tf.keras.models.load_model('models/ResNet-50-05-06-2024-15-07-29 he.h5')
model_cs1 = tf.keras.models.load_model('models/ResNet-50-05-10-2024-16-58-57cs2.h5')
model_cs2 = tf.keras.models.load_model('models/ResNet-50-05-10-2024-17-46-21cs.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image
        processed_filenames = {
            'original': filename,
            'he': 'processed_he_' + filename,
            'cs1': 'processed_cs1_' + filename,
            'cs2': 'processed_cs2_' + filename
        }
        
        processed_paths = {key: os.path.join(app.config['PROCESSED_FOLDER'], fname) for key, fname in processed_filenames.items()}
        
        histogram_equalization_rgb(file_path, processed_paths['he'])
        contrast_stretching_parabolic_rgb(file_path, processed_paths['cs1'], power=0.5)
        contrast_stretching_parabolic_rgb(file_path, processed_paths['cs2'], power=2)

        # Prepare the images for prediction
        img_original = image.load_img(file_path, target_size=(640, 640))
        img_he = image.load_img(processed_paths['he'], target_size=(640, 640))
        img_cs1 = image.load_img(processed_paths['cs1'], target_size=(640, 640))
        img_cs2 = image.load_img(processed_paths['cs2'], target_size=(640, 640))

        img_original = preprocess_input(np.expand_dims(image.img_to_array(img_original), axis=0))
        img_he = preprocess_input(np.expand_dims(image.img_to_array(img_he), axis=0))
        img_cs1 = preprocess_input(np.expand_dims(image.img_to_array(img_cs1), axis=0))
        img_cs2 = preprocess_input(np.expand_dims(image.img_to_array(img_cs2), axis=0))

        # Make predictions
        pred_original = model_original.predict(img_original)
        pred_he = model_he.predict(img_he)
        pred_cs1 = model_cs1.predict(img_cs1)
        pred_cs2 = model_cs2.predict(img_cs2)

        predictions = {
            'Original': pred_original,
            'HE': pred_he,
            'CS1': pred_cs1,
            'CS2': pred_cs2
        }

        # Determine predicted class and confidence for each model
        predicted_class = {key: np.argmax(pred, axis=1)[0] for key, pred in predictions.items()}
        confidence = {key: np.max(pred, axis=1)[0] for key, pred in predictions.items()}

        # Map the predicted class to the disease name
        class_names = ['bacterial leaf blight', 'brown spot', 'healthy', 'leaf blast', 'leaf scald', 'narrow brown spot']
        predicted_labels = {key: class_names[cls] if cls < len(class_names) else "Unknown" for key, cls in predicted_class.items()}

        # Set a confidence threshold (e.g., 0.6) to avoid false positives
        confidence_threshold = 0.6
        final_prediction = "No disease"
        for key, conf in confidence.items():
            if conf >= confidence_threshold:
                final_prediction = predicted_labels[key]
                break

        # Check if any prediction is "healthy" and if so, render no_disease.html
        if final_prediction == 'healthy':
            return render_template(
                'no_disease.html', 
                original_image=filename,
                processed_images=processed_filenames,
                accuracies={key: round(conf * 100, 2) for key, conf in confidence.items()}
            )

        # Generate graphs for each technique
        for key in predictions.keys():
            generate_graph(key, predictions[key])

        # Pass the accuracies to the template
        accuracies = {key: round(conf * 100, 2) for key, conf in confidence.items()}

        return render_template(
            'result.html', 
            original_image=filename, 
            processed_images=processed_filenames, 
            prediction=final_prediction,
            accuracies=accuracies  # Pass the accuracies to the template
        )

def generate_graph(technique, predictions):
    class_names = ['bacterial leaf blight', 'brown spot', 'healthy', 'leaf blast', 'leaf scald', 'narrow brown spot']
    probabilities = predictions[0] * 100

    fig, ax = plt.subplots()
    ax.barh(class_names, probabilities, color='skyblue')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title(f'Prediction Probability for {technique}')

    # Adjust margins and font size for better display of long category names
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(left=0.25)  # Adjust left margin to accommodate long category names
    for label in ax.get_yticklabels():
        label.set_fontsize(10)  # Set the font size for y-tick labels

    graph_path = os.path.join(app.config['GRAPHS_FOLDER'], f'{technique}_accuracy_graph.png')
    plt.savefig(graph_path)
    plt.close()


@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def send_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/get_graph/<technique>')
def get_graph(technique):
    filename = f'{technique}_accuracy_graph.png'
    response = make_response(send_from_directory(app.config['GRAPHS_FOLDER'], filename))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
@app.route('/details/<prediction>/<technique>')
def view_details(prediction, technique):
    techniques = {
        'original': 'Original',
        'he': 'Histogram Equalization',
        'cs1': 'Contrast Stretching 1',
        'cs2': 'Contrast Stretching 2'
    }
    
    technique_label = techniques.get(technique, 'Unknown Technique')

    return render_template(
        'details.html', 
        prediction=prediction, 
        technique=technique, 
        technique_label=technique_label, 
        processed_images=processed_images
    )

if __name__ == "__main__":
    app.run(debug=True)
