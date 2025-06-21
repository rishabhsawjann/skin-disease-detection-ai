import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  #epochs for more training
TRAIN_DIR = 'ModelTrain\SkinDiseaseDB\Train'

# Load the previously trained model
model = load_model('OUTPUT\optimized_model_final.h5')

# Unfreeze more layers to allow better fine-tuning
for layer in model.layers[-100:]:  # Unfreeze the last 100 layers
    layer.trainable = True

# Add L2 Regularization to Dense layers
for layer in model.layers:
    if isinstance(layer, Dense):
        layer.kernel_regularizer = l2(0.001)

# Compile the model with a slightly higher learning rate
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation with robust transformations
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    fill_mode='nearest'
)

# Load training and validation data
train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
)

# Callbacks for checkpoint, learning rate reduction, and early stopping
checkpoint = ModelCheckpoint(
    filepath='<ENTER THE PATH TO SAVE THE MODEL IN THE FOLLOWING FORMAT "PATH/FINALOUTPUT.keras">',
    monitor='val_accuracy', save_best_only=True, mode='max'
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Save the final fine-tuned model
model.save('<ENTER THE PATH TO SAVE THE MODEL IN THE FOLLOWING FORMAT "PATH/FINALOUTPUT.h5">')
print("Fine-tuned model saved successfully.")

