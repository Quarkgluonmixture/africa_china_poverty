"""
Saliency Map visualization script for ResNet18 model.
This script generates a saliency map (heatmap) to visualize which pixels
in a Chinese satellite image have the biggest impact on the model's prediction.
"""

import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import sys

# Add the models directory to the Python path
sys.path.append(os.path.dirname(__file__))

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
    
    return img_array, img  # Return both processed and original image

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

def compute_saliency_map(sess, processed_image, input_placeholder, output_tensor):
    """
    Compute the saliency map using gradients.
    
    Args:
        sess: tf.Session
        processed_image: np.array, preprocessed image with shape (1, 224, 224, 3)
        input_placeholder: tf.placeholder for input
        output_tensor: tf.Tensor, model output
    
    Returns:
        np.array: saliency map with shape (224, 224)
    """
    # Compute gradients of the output with respect to the input
    gradients = tf.gradients(output_tensor, input_placeholder)[0]
    
    # Run the session to get the gradients
    feed_dict = {input_placeholder: processed_image, 'is_training:0': False}
    grad_values = sess.run(gradients, feed_dict=feed_dict)
    
    # Take the absolute value of gradients
    grad_values = np.abs(grad_values)
    
    # Find the maximum across the channels (axis=3)
    saliency_map = np.max(grad_values, axis=3)[0]  # Shape: (224, 224)
    
    return saliency_map

def visualize_saliency(original_image, saliency_map, save_path='saliency_test.png'):
    """
    Visualize the original image, raw saliency map, and smoothed heatmap side by side.
    
    Args:
        original_image: PIL.Image, original image
        saliency_map: np.array, saliency map with shape (224, 224)
        save_path: str, path to save the visualization
    """
    # Apply Gaussian Blur to make it look like a heat map
    # 1. Slightly enhance contrast (optional)
    saliency_map = np.power(saliency_map, 2)
    
    # 2. Apply stronger Gaussian blur
    # Larger sigma values result in more blur and smoothing.
    # Try adjusting from 3 to 8-10 until the grid pattern disappears
    saliency_map_smoothed = gaussian_filter(saliency_map, sigma=10)
    
    # 3. Normalize to [0, 1] range
    saliency_map_smoothed = (saliency_map_smoothed - saliency_map_smoothed.min()) / (saliency_map_smoothed.max() - saliency_map_smoothed.min())
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot raw saliency map
    # Normalize the raw saliency map for better visualization
    norm_raw = Normalize(vmin=saliency_map.min(), vmax=saliency_map.max())
    heatmap_raw = cm.jet(norm_raw(saliency_map))
    
    # Overlay raw heatmap on the original image
    ax2.imshow(original_image)
    ax2.imshow(heatmap_raw, alpha=0.5, cmap='jet')
    ax2.set_title('Raw Saliency (Noisy)')
    ax2.axis('off')
    
    # Plot smoothed saliency map as heatmap
    # Normalize the smoothed saliency map for better visualization
    norm_smooth = Normalize(vmin=saliency_map_smoothed.min(), vmax=saliency_map_smoothed.max())
    heatmap_smooth = cm.jet(norm_smooth(saliency_map_smoothed))
    
    # Overlay smoothed heatmap on the original image with transparency
    ax3.imshow(original_image)
    ax3.imshow(heatmap_smooth, alpha=0.5, cmap='jet')
    ax3.set_title('Smoothed Heatmap')
    ax3.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saliency visualization saved to: {save_path}")

def main():
    # Set TensorFlow 1.x compatibility mode
    tf.compat.v1.disable_eager_execution()
    
    # Directory containing Chinese satellite images
    china_images_dir = os.path.join(os.path.dirname(__file__), 'china_dataset_final')
    
    # Create output directory for saliency maps
    output_dir = os.path.join(os.path.dirname(__file__), 'saliency_maps')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Check if directory exists
    if not os.path.exists(china_images_dir):
        print(f"Error: Directory {china_images_dir} does not exist!")
        return
    
    # Get all JPG files in the directory
    try:
        jpg_files = [f for f in os.listdir(china_images_dir) if f.lower().endswith('.jpg')]
        if not jpg_files:
            print(f"No JPG files found in {china_images_dir}")
            return
        print(f"Found {len(jpg_files)} images to process")
    except Exception as e:
        print(f"Error listing images: {e}")
        return
    
    # Create TensorFlow graph once to reuse for all images
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
            
            # Process each image
            for i, image_file in enumerate(jpg_files):
                print(f"\nProcessing image {i+1}/{len(jpg_files)}: {image_file}")
                
                # Get full path to the image
                image_path = os.path.join(china_images_dir, image_file)
                
                # Load and preprocess the image
                try:
                    processed_image, original_image = load_and_preprocess_image(image_path)
                    print(f"Image preprocessed successfully. Shape: {processed_image.shape}")
                except Exception as e:
                    print(f"Error preprocessing image: {e}")
                    continue
                
                # Compute saliency map
                try:
                    saliency_map = compute_saliency_map(sess, processed_image, X, output_tensor)
                    print(f"Saliency map computed successfully. Shape: {saliency_map.shape}")
                    
                    # Apply Gaussian smoothing with enhanced contrast and normalization
                    # Note: The actual processing is done in visualize_saliency function
                    print("Saliency map will be processed with enhanced contrast, stronger blur, and normalization")
                    
                    # Create output filename (remove .jpg extension and add _saliency.png)
                    output_filename = os.path.splitext(image_file)[0] + '_saliency.png'
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Visualize and save the results
                    visualize_saliency(original_image, saliency_map, output_path)
                    
                    print(f"Saved saliency map to: {output_path}")
                    
                except Exception as e:
                    print(f"Error computing saliency map: {e}")
                    continue
            
            print(f"\nSUCCESS: Processed {len(jpg_files)} images. Saliency maps saved to: {output_dir}")

if __name__ == "__main__":
    main()