import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare the validation generator
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

val_gen = val_datagen.flow_from_directory(
    '/home/ec2-user/ModelTrain/Skin cancer ISIC The International Skin Imaging Collaboration/Train',
    target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation'
)

# Save the current class mapping
with open('class_indices_from_validation.json', 'w') as f:
    json.dump(val_gen.class_indices, f)

print("Class indices from validation generator:", val_gen.class_indices)

