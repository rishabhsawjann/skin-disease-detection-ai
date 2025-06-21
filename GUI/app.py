from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import json
import os
import time  # For delay
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/sagarsoni/Desktop/SkinDiseaseAI/GUI/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the model and class indices
model = load_model('OUTPUT/finalv2optimized_model_final.h5')

with open('OUTPUT/class_indices_from_validation.json', 'r') as f:
    class_indices = json.load(f)

# Invert the dictionary to get index-to-class mapping
index_to_class = {v: k for k, v in class_indices.items()}

# Disease information dictionary with descriptions and cures
disease_info = {
    "actinic keratosis": {
        "description": "A rough, scaly patch on the skin caused by exposure to the sun.",
        "cure": "Use sunscreen and topical medications; freezing or surgery may be needed."
    },
    "basal cell carcinoma": {
        "description": "A type of skin cancer that begins in basal cells.",
        "cure": "Surgical removal or radiation therapy is recommended."
    },
    "Carcinoma": {
        "description": "A cancer that starts in the skin or the tissues lining other organs.",
        "cure": "Surgery, chemotherapy, or radiation therapy."
    },
    "Dermatitis": {
        "description": "Inflammation of the skin, causing itchiness and redness.",
        "cure": "Use moisturizing creams and corticosteroid creams."
    },
    "dermatofibroma": {
        "description": "A benign skin growth that can feel like a hard lump.",
        "cure": "Usually does not need treatment; surgical removal if bothersome."
    },
    "Eczema": {
        "description": "A condition that makes the skin red and itchy.",
        "cure": "Moisturizers, antihistamines, and corticosteroids can help."
    },
    "Keratosis": {
        "description": "A growth of keratin on the skin or mucous membranes.",
        "cure": "Topical creams or surgical removal if needed."
    },
    "Melanoma": {
        "description": "The most serious type of skin cancer.",
        "cure": "Surgery and immunotherapy are common treatments."
    },
    "Nevi": {
        "description": "Commonly known as moles, these are benign growths.",
        "cure": "Monitoring or surgical removal if necessary."
    },
    "nevus": {
        "description": "A type of birthmark or mole.",
        "cure": "Observation or removal if it changes in size or color."
    },
    "pigmented benign keratosis": {
        "description": "A harmless pigmented skin lesion.",
        "cure": "No treatment needed unless it causes discomfort."
    },
    "Psoriasis": {
        "description": "A chronic disease causing red, scaly patches on the skin.",
        "cure": "Topical treatments, phototherapy, or systemic medications."
    },
    "seborrheic keratosis": {
        "description": "A noncancerous skin growth, common in older adults.",
        "cure": "No treatment needed unless it irritates; removal if desired."
    },
    "Ringworm": {
        "description": "A fungal infection that causes a ring-shaped rash.",
        "cure": "Antifungal creams or medications are used."
    },
    "vascular lesion": {
        "description": "A type of birthmark made up of blood vessels.",
        "cure": "Laser therapy can be used for cosmetic reasons."
    },
    "squamous cell carcinoma": {
        "description": "A form of skin cancer that may spread to other organs.",
        "cure": "Surgery or radiation therapy is recommended."
    },
    "Warts": {
        "description": "Small growths on the skin caused by a viral infection.",
        "cure": "Topical treatments, freezing, or surgical removal."
    }
}

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_last_conv_layer(model):
    """Find the last convolutional layer in the model for Grad-CAM."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name or isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

def get_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    """Generate Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    return tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

def generate_grad_cam_image(img_path, heatmap, output_path):
    """Superimpose Grad-CAM heatmap on the original image and save."""
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = plt.cm.jet(heatmap)[:, :, :3]
    
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(output_path)

def generate_saliency_map(img_array, model, pred_index):
    """Generate Saliency Map."""
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, img_tensor)
    saliency_map = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    return (saliency_map - tf.reduce_min(saliency_map)) / (tf.reduce_max(saliency_map) - tf.reduce_min(saliency_map))

def predict_image(img_path):
    """Predict the class, generate Grad-CAM and Saliency Map images."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = index_to_class[predicted_index]

    # Grad-CAM
    last_conv_layer_name = find_last_conv_layer(model)
    heatmap = get_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_index=predicted_index)
    
    # Ensure paths are saved in the static/uploads directory
    grad_cam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'grad_cam.png')
    generate_grad_cam_image(img_path, heatmap, grad_cam_path)

    # Saliency Map
    saliency_map = generate_saliency_map(img_array, model, pred_index=predicted_index)
    saliency_map_path = os.path.join(app.config['UPLOAD_FOLDER'], 'saliency_map.png')
    plt.imsave(saliency_map_path, saliency_map, cmap='hot')

    return predicted_class, grad_cam_path, saliency_map_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in request'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            time.sleep(10)  # Optional delay

            predicted_class, grad_cam_path, saliency_map_path = predict_image(filepath)

            # Get disease info
            info = disease_info.get(predicted_class, {"description": "No description available.", "cure": "No cure information available."})

            return render_template(
                'index.html',
                prediction=predicted_class,
                description=info["description"],
                cure=info["cure"],
                grad_cam_url=url_for('static', filename=f'uploads/grad_cam.png'),
                saliency_map_url=url_for('static', filename=f'uploads/saliency_map.png'),
                filename=filename  # Pass filename to the template
            )
    return render_template('index.html', prediction=None, description=None, cure=None)

if __name__ == '__main__':
    app.run(debug=True)