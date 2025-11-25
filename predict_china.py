import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
from models.models_resnet import ResNet18

# Constants
CHINA_IMG_DIR = 'china_dataset_final'
IMG_SIZE = 224
CHECKPOINT_PATH = 'checkpoints/best_model.ckpt'
OUTPUT_CSV = 'china_predictions.csv'

def main():
    # 1. Setup and scan directory
    if not os.path.exists(CHINA_IMG_DIR):
        print(f"Warning: Directory '{CHINA_IMG_DIR}' not found. Please create it and add images.")
        # We proceed anyway so the script exists, but it won't find images yet.
        image_files = []
    else:
        image_files = [f for f in os.listdir(CHINA_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} images in {CHINA_IMG_DIR}")

    # 2. Build Graph (Same as training)
    tf.reset_default_graph()
    
    # Placeholders
    # Shape: [Batch Size, Height, Width, Channels]
    images_ph = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input_images')
    is_training_ph = tf.placeholder(tf.bool, name='is_training')
    
    # Build Model
    model = ResNet18(images_ph, num_outputs=1, is_training=is_training_ph)
    predictions_op = model.outputs
    
    # Saver to restore weights
    saver = tf.train.Saver()
    
    results = []
    
    # 3. Session and Inference
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        # Restore checkpoint
        try:
            if os.path.exists(CHECKPOINT_PATH + ".index") or os.path.exists(CHECKPOINT_PATH):
                saver.restore(sess, CHECKPOINT_PATH)
                print(f"Model restored from {CHECKPOINT_PATH}")
            else:
                print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
                return
        except Exception as e:
            print(f"Error restoring model: {e}")
            return

        print("\nStarting inference...")
        
        for img_file in image_files:
            # Parse filename
            # Format expectation: "0_LocationName.jpg" or "1_LocationName.jpg"
            try:
                # True Label is the first character
                true_label = int(img_file[0])
                
                # Location Name is the rest (stripping extension and prefix)
                base_name = os.path.splitext(img_file)[0]
                
                # Handle cases like "0_Name" vs "0Name"
                # We strip the first character. If the next is '_', we strip that too.
                location_name_raw = base_name[1:]
                if location_name_raw.startswith('_'):
                    location_name = location_name_raw[1:]
                else:
                    location_name = location_name_raw
                    
            except ValueError:
                print(f"Skipping file with unexpected format: {img_file}")
                continue

            img_path = os.path.join(CHINA_IMG_DIR, img_file)
            
            # --- Preprocessing ---
            # 1. Read Image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            
            # 2. Convert BGR to RGB (OpenCV uses BGR, TF model trained on RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Resize to 224x224
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # 4. Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # 5. Add batch dimension -> [1, 224, 224, 3]
            img_batch = np.expand_dims(img, axis=0)
            
            # --- Inference ---
            pred_val = sess.run(predictions_op, feed_dict={
                images_ph: img_batch, 
                is_training_ph: False
            })
            
            # Extract scalar prediction
            predicted_wealth = pred_val[0][0]
            
            results.append({
                'Image Name': img_file,
                'True Label': true_label,
                'Predicted Wealth Index': predicted_wealth,
                'Location Name': location_name
            })

    # 4. Output Results
    if results:
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved predictions to {OUTPUT_CSV}")
        
        # Sort by Predicted Wealth Index (Descending)
        df_sorted = df.sort_values(by='Predicted Wealth Index', ascending=False)
        
        print("\n--- Inference Results (Sorted by Predicted Wealth: Rich -> Poor) ---")
        # Adjust display options to show more columns if needed
        pd.set_option('display.max_rows', None)
        print(df_sorted[['Location Name', 'True Label', 'Predicted Wealth Index']].to_string(index=False))
        
        # Quick validation check
        print("\nSummary:")
        print(f"Total processed: {len(df)}")
        if 'True Label' in df.columns:
            avg_rich = df[df['True Label'] == 1]['Predicted Wealth Index'].mean()
            avg_poor = df[df['True Label'] == 0]['Predicted Wealth Index'].mean()
            print(f"Average Predicted Wealth for True Label 1 (Rich): {avg_rich:.4f}")
            print(f"Average Predicted Wealth for True Label 0 (Poor): {avg_poor:.4f}")
    else:
        print("No results generated. Check if images exist in the folder.")

if __name__ == '__main__':
    main()