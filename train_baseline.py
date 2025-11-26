"""
Baseline training script using VGG-F model for poverty prediction.
This script reproduces the baseline results from Jean et al. (2016).
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
import random
from sklearn.model_selection import train_test_split
from models.models_vgg import VGGF
import shutil

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

# Constants
DATA_DIR = 'data'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
CLUSTERS_PATH = os.path.join(DATA_DIR, 'clusters.csv')
CHECKPOINT_DIR = 'checkpoints'
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'baseline_vgg.ckpt')
IMAGE_SIZE = 224
CHANNELS = 3

def parse_args():
    parser = argparse.ArgumentParser(description='Train VGG-F for Poverty Prediction (Baseline)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    return parser.parse_args()

def check_image_exists(row):
    """Check if the image file exists for a given row (using unique_id or country+cluster_id)."""
    # 优先使用unique_id列
    if 'unique_id' in row:
        unique_id = row['unique_id']
    else:
        # 向后兼容：构建唯一ID
        unique_id = f"{row['country']}_{int(row['cluster_id'])}"
    
    img_path = os.path.join(IMAGES_DIR, f"{unique_id}.jpg")
    return os.path.exists(img_path)

def load_and_split_data():
    """
    Load data from CSV, filter missing images, and split into Train/Val/Test.
    """
    print(f"Loading data from {CLUSTERS_PATH}...")
    if not os.path.exists(CLUSTERS_PATH):
        raise FileNotFoundError(f"Could not find {CLUSTERS_PATH}. Please ensure data is prepared.")

    df = pd.read_csv(CLUSTERS_PATH)
    
    # 确保有unique_id列（如果没有则创建）
    if 'unique_id' not in df.columns:
        df['unique_id'] = df['country'] + '_' + df['cluster_id'].astype(int).astype(str)
    
    # Filter out missing images
    initial_count = len(df)
    df['image_exists'] = df.apply(check_image_exists, axis=1)
    df = df[df['image_exists']].drop(columns=['image_exists'])
    final_count = len(df)
    
    print(f"Found {final_count} clusters with images (filtered out {initial_count - final_count} missing).")

    # Split: Train (70%), Val (15%), Test (15%)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Data Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def _parse_function(filename, label, is_training):
    """
    TF Dataset map function to load and preprocess images with strong augmentation.
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    
    # Resize to ensure dimensions (though inputs are expected to be 224x224)
    image_resized = tf.image.resize_images(image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
    
    # Normalize to [0, 1]
    image_normalized = image_resized / 255.0
    
    # Strong Augmentation (only for training)
    if is_training:
        # Random horizontal and vertical flips
        image_normalized = tf.image.random_flip_left_right(image_normalized)
        image_normalized = tf.image.random_flip_up_down(image_normalized)
        
        # Random 90-degree rotation
        k = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
        image_normalized = tf.image.rot90(image_normalized, k=k)
        
        # Color jitter: brightness adjustment
        brightness_factor = tf.random_uniform([], minval=0.8, maxval=1.2)
        image_normalized = tf.clip_by_value(image_normalized * brightness_factor, 0.0, 1.0)
        
    return image_normalized, label

def create_dataset(df, batch_size, is_training):
    """
    Create a tf.data.Dataset from a pandas DataFrame.
    """
    # Create file paths using unique_id
    filenames = df['unique_id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")).values
    labels = df['wealth_index'].values.astype(np.float32)
    labels = np.expand_dims(labels, axis=1) # Shape [N, 1] for regression
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    # Map the parse function
    dataset = dataset.map(lambda f, l: _parse_function(f, l, is_training), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

def calculate_r2(y_true, y_pred):
    """Calculate R^2 score (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def main():
    args = parse_args()
    
    # Ensure checkpoint directory exists
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    # 1. Prepare Data
    train_df, val_df, test_df = load_and_split_data()
    
    # 2. Build Graph
    tf.reset_default_graph()
    
    # Placeholders
    is_training_ph = tf.placeholder(tf.bool, name='is_training')
    lr_ph = tf.placeholder(tf.float32, name='learning_rate')
    
    # Datasets and Iterators
    train_dataset = create_dataset(train_df, args.batch_size, is_training=True)
    val_dataset = create_dataset(val_df, args.batch_size, is_training=False)
    
    # Create an iterator structure that can handle the shapes/types of our datasets
    handle_ph = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle_ph, train_dataset.output_types, train_dataset.output_shapes)
    images, labels = iterator.get_next()
    
    # Build VGG-F Model (Baseline)
    print("Building VGG-F model (Baseline)...")
    model = VGGF(images, num_outputs=1, is_training=is_training_ph)
    predictions = model.outputs
    
    # Loss (MSE + Regularization)
    mse_loss = tf.losses.mean_squared_error(labels, predictions)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = mse_loss + tf.add_n(reg_losses) if reg_losses else mse_loss
    
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph)
    train_op = optimizer.minimize(total_loss)
        
    # Saver
    saver = tf.train.Saver(max_to_keep=1)
    
    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        # Create handles for the datasets
        train_iterator = train_dataset.make_initializable_iterator()
        val_iterator = val_dataset.make_initializable_iterator()
        
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())
        
        # Training Loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 20
        
        print(f"\nStarting VGG-F baseline training for {args.epochs} epochs...")
        print(f"Learning Rate: {args.learning_rate}, Batch Size: {args.batch_size}, Patience: {patience_limit}")
        
        for epoch in range(args.epochs):
            # --- Training Phase ---
            sess.run(train_iterator.initializer)
            train_losses = []
            
            try:
                while True:
                    _, batch_loss = sess.run([train_op, total_loss],
                                           feed_dict={handle_ph: train_handle,
                                                      is_training_ph: True,
                                                      lr_ph: args.learning_rate})
                    train_losses.append(batch_loss)
            except tf.errors.OutOfRangeError:
                pass
            
            avg_train_loss = np.mean(train_losses)
            
            # --- Validation Phase ---
            sess.run(val_iterator.initializer)
            val_preds = []
            val_labels = []
            val_losses = []
            
            try:
                while True:
                    pred_batch, label_batch, batch_loss = sess.run([predictions, labels, mse_loss],
                                                                 feed_dict={handle_ph: val_handle,
                                                                            is_training_ph: False})
                    val_preds.append(pred_batch)
                    val_labels.append(label_batch)
                    val_losses.append(batch_loss)
            except tf.errors.OutOfRangeError:
                pass
            
            val_preds = np.concatenate(val_preds, axis=0)
            val_labels = np.concatenate(val_labels, axis=0)
            avg_val_loss = np.mean(val_losses)
            
            # Calculate R^2
            val_r2 = calculate_r2(val_labels, val_preds)
            
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss (Total): {avg_train_loss:.4f} - "
                  f"Val MSE: {avg_val_loss:.4f} - "
                  f"Val R2: {val_r2:.4f}")
            
            # --- Checkpointing & Early Stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"  * New best model found! Saving to {BEST_MODEL_PATH}...")
                saver.save(sess, BEST_MODEL_PATH)
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience_limit}")
                
            if patience_counter >= patience_limit:
                print("\nEarly stopping triggered.")
                break
        
        print("\nTraining complete. Loading best model for final evaluation...")
        saver.restore(sess, BEST_MODEL_PATH)
        
        # --- Final Test Evaluation ---
        test_dataset = create_dataset(test_df, args.batch_size, is_training=False)
        test_iterator = test_dataset.make_initializable_iterator()
        test_handle = sess.run(test_iterator.string_handle())
        
        sess.run(test_iterator.initializer)
        test_preds = []
        test_labels = []
        test_losses = []
        
        try:
            while True:
                pred_batch, label_batch, batch_loss = sess.run([predictions, labels, mse_loss],
                                                             feed_dict={handle_ph: test_handle,
                                                                        is_training_ph: False})
                test_preds.append(pred_batch)
                test_labels.append(label_batch)
                test_losses.append(batch_loss)
        except tf.errors.OutOfRangeError:
            pass
        
        test_preds = np.concatenate(test_preds, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        avg_test_loss = np.mean(test_losses)
        test_r2 = calculate_r2(test_labels, test_preds)
        
        print(f"\n=== Final Test Results ===")
        print(f"Test MSE: {avg_test_loss:.4f}")
        print(f"Test R2: {test_r2:.4f}")
        
        # Save predictions
        test_df_copy = test_df.copy()
        test_df_copy['predicted_wealth'] = test_preds.flatten()
        test_df_copy['actual_wealth'] = test_labels.flatten()
        predictions_path = 'baseline_predictions.csv'
        test_df_copy[['unique_id', 'actual_wealth', 'predicted_wealth']].to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

if __name__ == '__main__':
    main()