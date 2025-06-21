import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import json

# Load the class indices from the JSON file
with open('/Users/sagarsoni/Downloads/SkinDiseaseAI/OUTPUT/class_indices_from_validation.json', 'r') as f:
    class_indices = json.load(f)

# Invert the dictionary to get index-to-class mapping
index_to_class = {v: k for k, v in class_indices.items()}
print("Index-to-class mapping:", index_to_class)

# Load the trained model
model = load_model('/Users/sagarsoni/Downloads/SkinDiseaseAI/OUTPUT/finalv2optimized_model_final.h5')

# Function to find the last convolutional layer in the model
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        # Check if the layer is a convolutional layer
        if 'conv' in layer.name or isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

# Get the name of the last convolutional layer
last_conv_layer_name = find_last_conv_layer(model)
print("Last convolutional layer:", last_conv_layer_name)

# Path to the image for prediction
img_path = r'/Users/sagarsoni/Downloads/SkinDiseaseAI/ModelTrain/SkinDiseaseDB/Train/Eczema/0_0.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions, axis=1)[0]
predicted_class = index_to_class[predicted_index]  # Map the predicted index to the class label

# Function to generate Grad-CAM heatmap
def get_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Forward pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Generate heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Generate Grad-CAM heatmap
heatmap = get_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_index=predicted_index)

# Function to superimpose heatmap on the original image
def superimpose_heatmap(img_path, heatmap, alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to 0-255
    heatmap = np.resize(heatmap, (img.shape[0], img.shape[1]))  # Resize heatmap to match image size

    # Apply colormap
    heatmap = plt.cm.jet(heatmap)[:, :, :3]
    superimposed_img = heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# Superimpose and display Grad-CAM heatmap on image
superimposed_img = superimpose_heatmap(img_path, heatmap)
plt.imshow(superimposed_img)
plt.title(f'Grad-CAM Prediction: {predicted_class}')
plt.axis('off')
plt.show()

# Function to generate Saliency Map
def get_saliency_map(img_array, model, pred_index):
    # Convert the image array to a tensor and enable gradient tracking
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        # Use the index of the predicted class or specify one
        class_channel = predictions[:, pred_index]

    # Get the gradient of the predicted class with respect to the input image
    grads = tape.gradient(class_channel, img_tensor)
    # Take the maximum gradient across the color channels
    saliency_map = tf.reduce_max(tf.abs(grads), axis=-1)[0]  # Keep only the first example in the batch

    # Normalize the saliency map to the range [0, 1]
    saliency_map = (saliency_map - tf.reduce_min(saliency_map)) / (tf.reduce_max(saliency_map) - tf.reduce_min(saliency_map))
    return saliency_map.numpy()

# Generate and display Saliency Map
saliency_map = get_saliency_map(img_array, model, pred_index=predicted_index)
plt.imshow(saliency_map, cmap='hot')
plt.title('Saliency Map')
plt.axis('off')
plt.show()