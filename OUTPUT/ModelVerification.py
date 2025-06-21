from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
VALIDATION_DIR = '/home/ec2-user/ModelTrain/Skin cancer ISIC The International Skin Imaging Collaboration/Train'

# Step 1: Load the saved .h5 model
model_path = "/home/ec2-user/ModelTrain/OUTPUT/optimized_model_final.h5"
model = load_model(model_path)

# Step 2: Prepare validation data generator
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # Use the same split as before
)

# Load the validation data generator and print class mapping
val_gen = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Print the class-to-index mapping
print("Class-to-index mapping from validation generator:")
print(val_gen.class_indices)

# Invert the class-to-index mapping to index-to-class for easy lookup
index_to_class = {v: k for k, v in val_gen.class_indices.items()}
print("\nIndex-to-class mapping:")
print(index_to_class)

# Step 3: Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Step 4: Verify that the model outputs match the class indices order
print("\nVerifying model output layer shape:")
print(f"Model output shape: {model.output.shape}")  # Should match the number of classes

