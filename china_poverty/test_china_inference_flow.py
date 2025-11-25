"""
Test script to verify ResNet18 model can process Chinese satellite images.
This script loads a random image from china_dataset_final directory and passes it
through the ResNet18 model to verify the inference flow works correctly.
"""

import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import sys

# Add the models directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'africa_poverty'))

# Import ResNet18 model
from models.models_resnet import ResNet18

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for ResNet18 model.
    
    Args:
        image_path: str, path to the image file
        target_size: tuple, target size (height, width)
    
    Returns:
        np.array: preprocessed image with shape (1, 224, 224, 3)
    """
    # Load image using PIL
    img = Image.open(image_path)
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension: shape becomes (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_random_image_from_directory(directory):
    """
    Get a random JPG image from the specified directory.
    
    Args:
        directory: str, path to the directory containing images
    
    Returns:
        str: path to a randomly selected image file
    """
    # Get all JPG files in the directory
    jpg_files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
    
    if not jpg_files:
        raise ValueError(f"No JPG files found in {directory}")
    
    # Select a random image
    random_image = random.choice(jpg_files)
    return os.path.join(directory, random_image)

def main():
    # Set TensorFlow 1.x compatibility mode
    tf.compat.v1.disable_eager_execution()
    
    # Directory containing Chinese satellite images
    china_images_dir = os.path.join(os.path.dirname(__file__), 'china_dataset_final')
    
    # Check if directory exists
    if not os.path.exists(china_images_dir):
        print(f"Error: Directory {china_images_dir} does not exist!")
        return
    
    # Get a random image
    try:
        image_path = get_random_image_from_directory(china_images_dir)
        print(f"Selected image: {os.path.basename(image_path)}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load and preprocess the image
    try:
        processed_image = load_and_preprocess_image(image_path)
        print(f"Image preprocessed successfully. Shape: {processed_image.shape}")
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return
    
    # Create TensorFlow graph
    with tf.compat.v1.Graph().as_default():
        # Create placeholder for input
        X = tf.compat.v1.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
        
        # Create is_training placeholder
        is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        
        # Instantiate ResNet18 model
        try:
            model = ResNet18(inputs=X, num_outputs=1, is_training=is_training)
            print("ResNet18 model instantiated successfully")
        except Exception as e:
            print(f"Error instantiating ResNet18 model: {e}")
            return
        
        # Get the output tensor
        output_tensor = model.outputs
        print(f"Output tensor shape: {output_tensor.shape}")
        
        # Create session and run inference
        with tf.compat.v1.Session() as sess:
            # Initialize variables
            try:
                sess.run(tf.compat.v1.global_variables_initializer())
                print("Global variables initialized successfully")
            except Exception as e:
                print(f"Error initializing variables: {e}")
                return
            
            # Run inference
            try:
                feed_dict = {X: processed_image, is_training: False}
                output_value = sess.run(output_tensor, feed_dict=feed_dict)
                
                print("\n--- Inference Results ---")
                print(f"Output Value: {output_value}")
                print(f"Output Shape: {output_value.shape}")
                
                # Success message
                print("\nSUCCESS: China Image successfully passed through ResNet-18!")
                
            except Exception as e:
                print(f"Error during inference: {e}")
                return

if __name__ == "__main__":
    main()