import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
dataset_dir = 'SkinDiseaseDB/Train'
output_dir = 'SkinDiseaseDBSplit'
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# Create output directories for train and test if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Set the split ratio
test_size = 0.2  # 20% of data for testing

# Iterate through each class folder in the dataset directory
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):  # Ensure it's a directory
        # Get list of all images in the class folder
        images = [img for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split images into train and test
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        
        # Create directories for each class in the train and test folders
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Move training images to the train directory
        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(train_class_dir, img)
            shutil.copy2(src_path, dst_path)
        
        # Move testing images to the test directory
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(test_class_dir, img)
            shutil.copy2(src_path, dst_path)

print("Dataset split completed. Training and testing sets are saved in 'SkinDiseaseDBSplit' directory.")