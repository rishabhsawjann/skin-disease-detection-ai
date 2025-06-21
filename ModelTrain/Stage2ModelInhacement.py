import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 16
TRAIN_DIR = '/Users/sagarsoni/Desktop/SkinDisese AI/Skin cancer ISIC The International Skin Imaging Collaboration/Train'

# Load the previously trained model
model = load_model('/Users/sagarsoni/Desktop/SkinDisese_AI/final_fine_tuned_model.h5')

# Unfreeze some of the last layers of the base model (ResNet50)
for layer in model.layers[-50:]:  # Unfreeze the last 50 layers for fine-tuning
    layer.trainable = True

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of the data for validation
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.2
)

# Load training and validation data
train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Model checkpoint to save the best model
checkpoint = ModelCheckpoint(
    filepath='/Users/sagarsoni/Desktop/SkinDisese_AI/NEW3_best_fine_tuned_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

# Continue training the model with callbacks (no early stopping)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,  # Additional epochs for fine-tuning
    callbacks=[checkpoint, reduce_lr]  # Removed early_stop
)

# Save the final fine-tuned model
model.save('/Users/sagarsoni/Desktop/SkinDisese_AI/NEW3_best_fine_tuned_model.h5')
print("Fine-tuned model saved successfully.")

# Plotting the training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()