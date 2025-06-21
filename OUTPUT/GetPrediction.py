import matplotlib.pyplot as plt
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the class indices from the JSON file
with open('/Users/sagarsoni/Downloads/SkinDiseaseAI/OUTPUT/class_indices_from_validation.json', 'r') as f:
    class_indices = json.load(f)

# Invert the dictionary to get index-to-class mapping
index_to_class = {v: k for k, v in class_indices.items()}
print("Index-to-class mapping:", index_to_class)

# Load the trained model
model = load_model('/Users/sagarsoni/Downloads/SkinDiseaseAI/OUTPUT/finalv2optimized_model_final.h5')

# Path to the image for prediction
img_path = r'/Users/sagarsoni/Downloads/SkinDiseaseAI/ModelTrain/SkinDiseaseDB/Train/Eczema/0_0.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions, axis=1)[0]

# Map the predicted index to the class label
predicted_class = index_to_class[predicted_index]  # Corrected: use int key

# Display the image and prediction result
plt.imshow(image.load_img(img_path))  # Display the image
plt.title(f'Predicted class: {predicted_class}')  # Display prediction as title
plt.axis('off')  # Turn off axis labels
plt.show()