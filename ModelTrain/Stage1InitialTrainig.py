import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
TRAIN_DIR = '/Users/sagarsoni/Desktop/SkinDisese AI/Skin cancer ISIC The International Skin Imaging Collaboration/Train'

print("Setting up data generators...")

# Data preparation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% of the data for validation
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

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

print("Data generators ready.")

# Model setup with ResNet50 pretrained on ImageNet
print("Setting up ResNet50 model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freezing the ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# Adding custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling instead of flattening
x = Dense(1024, activation='relu')(x)  # Adding a fully connected layer
x = Dropout(0.5)(x)  # Dropout layer to avoid overfitting
predictions = Dense(train_gen.num_classes, activation='softmax')(x)  # Output layer with softmax

# Defining the model
model = Model(inputs=base_model.input, outputs=predictions)

print("Model setup complete.")

# Compile the model
print("Compiling model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled.")

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Checkpoint to save the best model with .keras extension
checkpoint = ModelCheckpoint(
    filepath='/Users/sagarsoni/Desktop/SkinDisese_AI/best_skin_disease_model.keras',  # Use .keras extension
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Training the model with callbacks
print("Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print("Training complete.")

# Save the final model manually in .h5 format
print("Saving final model...")
model.save('/Users/sagarsoni/Desktop/SkinDisese_AI/final_skin_disease_model.h5')
print("Final model saved successfully.")

# Plotting the training and validation accuracy
print("Plotting accuracy...")
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting the training and validation loss
print("Plotting loss...")
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()